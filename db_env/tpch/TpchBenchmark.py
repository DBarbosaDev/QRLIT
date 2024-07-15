from datetime import datetime
from typing import Tuple, List

import numpy as np
from threading import Thread

from db_env.Benchmark import Benchmark
from db_env.tpch.TpchGenerator import TpchGenerator
from db_env.tpch.config import MAX_REFRESH_FILE_INDEX, STREAM_COUNT, SCALE_FACTOR, DB_REFRESH_ID
from db_env.tpch.tpch_stream.QueryStream import QueryStream
from db_env.tpch.tpch_stream.RefreshPair import RefreshPair
from db_env.tpch.tpch_stream.RefreshStream import RefreshStream
from db_env.tpch.tpch_stream.Stream import Stream
from shared_utils.utils import create_logger, get_connection


max_rand = 628163840
random_gen = np.random.default_rng(max_rand)

class TpchBenchmark(Benchmark):
    def __init__(self):
        self._log = create_logger('tpch_benchmark')
        self._load_rf_id()

        self.throughput_total_execution_time_seconds = 0
        self.power_total_queries_execution_time_seconds = 0
        self.power_total_refresh_functions_execution_time_seconds = 0
        self.benchmark_metrics_raw_results = {'power_metric': dict(), 'throughput_metric_streams': [] }

        self.query_gen_seeds = random_gen.integers(high=max_rand, low=1, size=6000)
        self.seed_index = 0

    def prepare_queries(self) -> None:
        generator = TpchGenerator()
        # mmddhhmmss

        # This always generates random seed at each execution. So to compare the classic and hybrid, the hybrid should also run using the same seeds
        # seed = int(datetime.now().strftime("%m%d%H%M%S"))
        
        # This will use the seed based on scale factor for all the algorithm executions.
        seed = self._get_query_gen_seed()

        generator.generate_queries(seed, STREAM_COUNT + 1)

    def execute(self) -> float:
        self.prepare_queries()

        power_size = self._run_power_test()
        self._inc_refresh_file_index()

        throughput_size = self._run_throughput_test()
        self._inc_refresh_file_index(STREAM_COUNT)

        self._save_rf_id()

        # QphH
        return (power_size * throughput_size) ** (1 / 2)

    def get_execution_times(self) -> tuple[float, float, float]:
        return (self.power_total_queries_execution_time_seconds, self.power_total_refresh_functions_execution_time_seconds, self.throughput_total_execution_time_seconds)
    
    def get_benchmark_metrics_raw_data(self) -> dict:
        return self.benchmark_metrics_raw_results

    def _get_query_gen_seed(self) -> int:
        seed = self.query_gen_seeds[self.seed_index]
        self.seed_index += 1
        return seed

    def _inc_refresh_file_index(self, number: int = 1):
        # modulo but not zero (i + m - 1) % n + 1
        self._refresh_file_index = (self._refresh_file_index + number - 1) % MAX_REFRESH_FILE_INDEX + 1

    def _load_rf_id(self):
        with open(DB_REFRESH_ID, 'r') as f:
            for line in f:
                self._refresh_file_index = int(line)

    def _save_rf_id(self):
        with open(DB_REFRESH_ID, 'w') as f:
            f.write(str(self._refresh_file_index))

    def _run_power_test(self) -> float:
        """
        Execute refresh pair 1, query stream and refresh pair 2 in sequence to
        measure database benchmark for single user and saves measurements.

        :return: calculated power size measure
        """

        refresh_pair, query_stream = self._prepare_power_test()
        return self._execute_power_test(refresh_pair, query_stream)

    def _prepare_power_test(self) -> Tuple[RefreshPair, QueryStream]:
        connection, cursor = get_connection(self._log, True)
        refresh_pair = RefreshPair('refresh_pair_powertest', self._refresh_file_index, connection, cursor)
        query_stream = QueryStream('query_stream_powertest', 0, connection, cursor)
        refresh_pair.load_data()
        query_stream.load_data()

        return refresh_pair, query_stream

    def _execute_power_test(self, refresh_pair: RefreshPair, query_stream: QueryStream) -> float:
        refresh_pair.execute_refresh_function1()
        query_stream.execute_stream()
        refresh_pair.execute_refresh_function2()

        return self._calculate_power_test_result(refresh_pair, query_stream)

    def _calculate_power_test_result(self, refresh_pair: RefreshPair, query_stream: QueryStream) -> float:
        # data object of stream.df_measures (Q = Query = QI, RF = Refresh Function = RI): 
        #      name       ,          time
        #   Q (number)    , 0 days 00:00:00.027785
        #   RF(number)    , 0 days 00:00:00.027785

        # Contains all the executions times for QI and RI based on object described in comment below
        power_test_results = query_stream.df_measures.append(refresh_pair.df_measures)

        # converts all QI and RI execution times to seconds
        time_results_in_seconds = power_test_results['time'].apply(lambda x: x.total_seconds())

        # multiply all the queris and refresh functions execution times
        multiplied_results = np.prod(time_results_in_seconds)

        # multiplied results powered to 1/24 which is the same as 24th root.
        geometric_mean = multiplied_results ** (1 / 24)

        power_size = 3600 * SCALE_FACTOR / geometric_mean

        # Collect power metric data
        self.benchmark_metrics_raw_results['power_metric'] = time_results_in_seconds.to_dict()
        self.power_total_queries_execution_time_seconds = query_stream.df_measures['time'].apply(lambda x: x.total_seconds()).sum()
        self.power_total_refresh_functions_execution_time_seconds = refresh_pair.df_measures['time'].apply(lambda x: x.total_seconds()).sum()

        return power_size

    def _run_throughput_test(self) -> float:
        """
        Execute refresh stream and given number of query streams in parallel threads and
        measure database benchmark for single user and saves measurements.

        :return: calculated throughput size measure
        """

        streams, processes = self._prepare_throughput_test()
        return self._execute_throughput_test(streams, processes)

    def _prepare_throughput_test(self) -> Tuple[List[Stream], List[Thread]]:
        processes = []
        streams = [RefreshStream('refresh_stream', STREAM_COUNT, self._refresh_file_index)]
        for i in range(STREAM_COUNT):
            streams.append(QueryStream(f'query_stream_{i + 1}', i + 1))

        for stream in streams:
            stream.load_data()
            processes.append(Thread(target=stream.execute_stream))

        return streams, processes

    def _execute_throughput_test(self, streams: List[Stream], processes: List[Thread]) -> float:
        # Execute streams in parallel threads
        for execute_stream_process in processes:
            execute_stream_process.start()

        # Wait for all processes to end
        for proc in processes:
            proc.join()

        return self._calculate_throughput_test_result(streams)

    def _calculate_throughput_test_result(self, streams: List[Stream]):
        # data object of stream.df_measures (Q = Query): 
        #     name       ,          time
        #   Q(number)    , 0 days 00:00:00.027785

        # The total amount of time is the stream which takes longer to execute
        total_time = max(stream.df_measures['time'].sum() for stream in streams)
        throughput_size = (len(streams) - 1) * 22 * 3600 * SCALE_FACTOR / total_time.total_seconds()

        # Collect power metric data
        self.benchmark_metrics_raw_results['throughput_metric_streams'] = []
        for stream in streams:
            self.benchmark_metrics_raw_results['throughput_metric_streams'].append(stream.df_measures['time'].apply(lambda x: x.total_seconds()).to_dict())
        self.throughput_total_execution_time_seconds = total_time.total_seconds()

        return throughput_size
