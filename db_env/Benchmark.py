from abc import ABC, abstractmethod


class Benchmark(ABC):
    @abstractmethod
    def prepare_queries(self) -> None:
        """Render queries to be executed during benchmark."""
        raise NotImplementedError

    @abstractmethod
    def execute(self) -> float:
        """:return: QphH metrics measured during executed benchmark."""
        raise NotImplementedError
    
    @abstractmethod
    def get_execution_times(self) -> tuple[float, float, float]:
        """
        :info: Q = Query = QI, RF = Refresh Function = RI, Ts = Total Execution Time
        :units: Seconds
        :return: (Power: sum all QI exec times, Power: sum all RI exec times, Throughput: Ts) 
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_benchmark_metrics_raw_data(self) -> dict:
        raise NotImplementedError
