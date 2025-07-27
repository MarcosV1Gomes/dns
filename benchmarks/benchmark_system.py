"""
Sistema de Benchmarks para Comparação de Estruturas DNS
=======================================================

Este módulo implementa um sistema abrangente de benchmarks para comparar
o desempenho das árvores AVL e Rubro-Negra em cenários diferentes.
"""

import sys
import os
import time
import random
import json
import gc
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Adicionar diretórios ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_structures'))

from dns_system import DNSSystem, TreeType
from dns_data_generator import DNSDataGenerator


class BenchmarkType(Enum):
    """Tipos de benchmark disponíveis."""
    DNS_REALISTIC = "dns_realistic"
    INSERT_ONLY = "insert_only"
    SEARCH_ONLY = "search_only"
    DELETE_ONLY = "delete_only"



@dataclass
class BenchmarkConfig:
    """Configuração para um benchmark."""
    name: str
    data_size: int
    operation_count: int
    benchmark_type: BenchmarkType
    operation_distribution: Dict[str, float]
    seed: int = 42


@dataclass
class BenchmarkRun:
    """Resultado de uma execução individual de benchmark."""
    total_time: float
    operations_per_second: float
    tree_height: int
    operation_times: Dict[str, List[float]]
    operation_stats: Dict[str, Dict[str, float]]
    tree_stats: Dict[str, Any]


@dataclass
class BenchmarkResult:
    """Resultado agregado de múltiplas execuções de benchmark."""
    config: BenchmarkConfig
    tree_type: TreeType
    runs: List[BenchmarkRun]
    num_runs: int
    warmup_runs: int
    
    # Estatísticas agregadas
    mean_total_time: float
    std_total_time: float
    mean_operations_per_second: float
    std_operations_per_second: float
    mean_tree_height: float
    std_tree_height: float
    
    # Estatísticas de operações agregadas
    aggregated_operation_stats: Dict[str, Dict[str, float]]


class DNSBenchmarkSuite:
    """
    Suite de benchmarks para comparação de estruturas DNS.
    
    Executa testes abrangentes em diferentes cenários e coleta
    métricas detalhadas de desempenho.
    """
    
    def __init__(self, seed: int = 42, num_runs: int = 5, warmup_runs: int = 2):
        """
        Inicializa a suite de benchmarks.
        
        Args:
            seed: Semente para reprodutibilidade
            num_runs: Número de execuções para cada benchmark
            warmup_runs: Número de execuções de aquecimento (descartadas)
        """
        self.seed = seed
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        random.seed(seed)
        
        # Configurações padrão de benchmark
        self.default_configs = self._create_default_configs()
        
        # Resultados dos benchmarks
        self.results: List[BenchmarkResult] = []
    
    def _create_default_configs(self) -> List[BenchmarkConfig]:
        """Cria configurações padrão de benchmark."""
        configs = []
        
        # Tamanhos de dados para teste
        data_sizes = [100, 1000, 10000]
        
        # Cenário DNS realista
        dns_distribution = {
            'search': 0.90,
            'insert': 0.05,
            'add_ip': 0.03,
            'delete': 0.02
        }
        
        for size in data_sizes:
            # Cenário DNS realista
            configs.append(BenchmarkConfig(
                name=f"DNS_Realistic_{size}",
                data_size=size,
                operation_count=size * 2,  # 2x operações do que dados
                benchmark_type=BenchmarkType.DNS_REALISTIC,
                operation_distribution=dns_distribution
            ))
            
            # Teste de inserção pura
            configs.append(BenchmarkConfig(
                name=f"Insert_Only_{size}",
                data_size=size,
                operation_count=size,
                benchmark_type=BenchmarkType.INSERT_ONLY,
                operation_distribution={'insert': 1.0}
            ))
            
            # Teste de busca pura
            configs.append(BenchmarkConfig(
                name=f"Search_Only_{size}",
                data_size=size,
                operation_count=size * 3,  # 3x buscas
                benchmark_type=BenchmarkType.SEARCH_ONLY,
                operation_distribution={'search': 1.0}
            ))
            
            # Teste de remoção pura
            configs.append(BenchmarkConfig(
                name=f"Delete_Only_{size}",
                data_size=size,
                operation_count=size // 2,  # Remover metade
                benchmark_type=BenchmarkType.DELETE_ONLY,
                operation_distribution={'delete': 1.0}
            ))
            

        
        return configs
    
    def run_benchmark(self, config: BenchmarkConfig, tree_type: TreeType) -> BenchmarkResult:
        """
        Executa um benchmark específico com múltiplas execuções e warmup.
        
        Args:
            config: Configuração do benchmark
            tree_type: Tipo de árvore a testar
            
        Returns:
            Resultado agregado do benchmark
        """
        print(f"Executando {config.name} com {tree_type.value} ({self.num_runs} execuções + {self.warmup_runs} warmup)...")
        
        all_runs = []
        
        # Executar warmup + execuções reais
        total_executions = self.warmup_runs + self.num_runs
        
        for run_idx in range(total_executions):
            is_warmup = run_idx < self.warmup_runs
            run_name = "warmup" if is_warmup else f"run {run_idx - self.warmup_runs + 1}"
            
            print(f"  {run_name}...", end=" ", flush=True)
            
            # Executar uma execução individual
            run_result = self._execute_single_run(config, tree_type, run_idx)
            
            # Só armazenar execuções não-warmup
            if not is_warmup:
                all_runs.append(run_result)
            
            print(f"✓ {run_result.operations_per_second:.0f} ops/s")
        
        # Calcular estatísticas agregadas
        return self._aggregate_results(config, tree_type, all_runs)
    
    def _execute_single_run(self, config: BenchmarkConfig, tree_type: TreeType, run_idx: int) -> BenchmarkRun:
        """
        Executa uma única execução de benchmark.
        
        Args:
            config: Configuração do benchmark
            tree_type: Tipo de árvore
            run_idx: Índice da execução (para variação de seed)
            
        Returns:
            Resultado da execução individual
        """
        # Forçar garbage collection antes do teste
        gc.collect()
        
        # Usar seed diferente para cada execução para variação
        run_seed = self.seed + run_idx * 1000
        
        # Criar sistema DNS
        dns_system = DNSSystem(tree_type)
        
        # Gerar dados iniciais se necessário
        if config.data_size > 0:
            generator = DNSDataGenerator(run_seed)
            initial_data = generator.generate_domain_ip_pairs(config.data_size)
            
            for domain, ips in initial_data:
                dns_system.add_domain(domain, ips[0])
                for ip in ips[1:]:
                    dns_system.add_ip_to_domain(domain, ip)
        
        # Resetar estatísticas após carregamento inicial
        dns_system.reset_stats()
        
        # Gerar operações de teste
        operations = self._generate_operations(config, dns_system, run_seed + 100)
        
        # Inicializar métricas de tempo e operações
        operation_times = {op_type: [] for op_type in config.operation_distribution.keys()}
        start_time = time.perf_counter()
        
        for operation in operations:
            op_start = time.perf_counter()
            
            if operation['type'] == 'search':
                dns_system.search_domain(operation['domain'])
            elif operation['type'] == 'insert':
                dns_system.add_domain(operation['domain'], operation['ip'])
            elif operation['type'] == 'add_ip':
                dns_system.add_ip_to_domain(operation['domain'], operation['ip'])
            elif operation['type'] == 'delete':
                dns_system.remove_domain(operation['domain'])
            
            op_end = time.perf_counter()
            operation_times[operation['type']].append(op_end - op_start)
        
        end_time = time.perf_counter()
        
        # Calcular métricas
        total_time = end_time - start_time
        operations_per_second = config.operation_count / total_time
        tree_height = dns_system.get_tree_height()
        
        # Calcular estatísticas de operações
        operation_stats = {}
        for op_type, times in operation_times.items():
            if times:
                # Converter tempos para milissegundos
                times_ms = [t * 1000 for t in times]
                operation_stats[op_type] = {
                    'count': len(times_ms),
                    'total_time': sum(times_ms),
                    'avg_time': sum(times_ms) / len(times_ms),
                    'min_time': min(times_ms),
                    'max_time': max(times_ms),
                    'std_dev': self._calculate_std_dev(times_ms)
                }
        
        # Obter estatísticas da árvore
        tree_stats = dns_system.get_system_stats()
        # Garantir que os campos de rotação estejam presentes
        if 'left_rotations' not in tree_stats:
            tree_stats['left_rotations'] = 0
        if 'right_rotations' not in tree_stats:
            tree_stats['right_rotations'] = 0
        if 'total_rotations' not in tree_stats:
            tree_stats['total_rotations'] = tree_stats['left_rotations'] + tree_stats['right_rotations']
        return BenchmarkRun(
            total_time=total_time,
            operations_per_second=operations_per_second,
            tree_height=tree_height,
            operation_times=operation_times,
            operation_stats=operation_stats,
            tree_stats=tree_stats
        )
    
    def _aggregate_results(self, config: BenchmarkConfig, tree_type: TreeType, runs: List[BenchmarkRun]) -> BenchmarkResult:
        """
        Agrega resultados de múltiplas execuções.
        
        Args:
            config: Configuração do benchmark
            tree_type: Tipo de árvore
            runs: Lista de execuções individuais
            
        Returns:
            Resultado agregado
        """
        if not runs:
            raise ValueError("Nenhuma execução válida para agregar")
        
        # Calcular estatísticas básicas
        total_times = [run.total_time for run in runs]
        ops_per_second = [run.operations_per_second for run in runs]
        tree_heights = [run.tree_height for run in runs]
        
        # Agregar estatísticas de operações
        aggregated_op_stats = {}
        all_op_types = set()
        for run in runs:
            all_op_types.update(run.operation_stats.keys())
        
        for op_type in all_op_types:
            op_avg_times = []
            op_counts = []
            op_total_times = []
            
            for run in runs:
                if op_type in run.operation_stats:
                    op_avg_times.append(run.operation_stats[op_type]['avg_time'])
                    op_counts.append(run.operation_stats[op_type]['count'])
                    op_total_times.append(run.operation_stats[op_type]['total_time'])
            
            if op_avg_times:
                aggregated_op_stats[op_type] = {
                    'mean_avg_time': sum(op_avg_times) / len(op_avg_times),
                    'std_avg_time': self._calculate_std_dev(op_avg_times),
                    'mean_count': sum(op_counts) / len(op_counts),
                    'mean_total_time': sum(op_total_times) / len(op_total_times),
                    'min_avg_time': min(op_avg_times),
                    'max_avg_time': max(op_avg_times)
                }
        
        return BenchmarkResult(
            config=config,
            tree_type=tree_type,
            runs=runs,
            num_runs=len(runs),
            warmup_runs=self.warmup_runs,
            mean_total_time=sum(total_times) / len(total_times),
            std_total_time=self._calculate_std_dev(total_times),
            mean_operations_per_second=sum(ops_per_second) / len(ops_per_second),
            std_operations_per_second=self._calculate_std_dev(ops_per_second),
            mean_tree_height=sum(tree_heights) / len(tree_heights),
            std_tree_height=self._calculate_std_dev(tree_heights),
            aggregated_operation_stats=aggregated_op_stats
        )
    
    def _generate_operations(self, config: BenchmarkConfig, dns_system: DNSSystem, seed: int = None) -> List[Dict[str, str]]:
        """Gera lista de operações para o benchmark."""
        if seed is not None:
            random.seed(seed)
        
        operations = []
        generator = DNSDataGenerator(seed or self.seed)
        
        # Obter domínios existentes para operações de busca/remoção
        existing_domains = [domain for domain, _ in dns_system.list_all_domains()]
        
        for _ in range(config.operation_count):
            # Escolher tipo de operação baseado na distribuição
            op_type = self._choose_operation_type(config.operation_distribution)
            
            if op_type == 'search':
                if existing_domains:
                    domain = random.choice(existing_domains)
                else:
                    domain = generator.generate_realistic_domain()
                operations.append({'type': 'search', 'domain': domain})
            
            elif op_type == 'insert':
                domain = generator.generate_realistic_domain()
                ip = generator.generate_realistic_ip()
                operations.append({'type': 'insert', 'domain': domain, 'ip': ip})
                existing_domains.append(domain)  # Adicionar à lista para futuras operações
            
            elif op_type == 'add_ip':
                if existing_domains:
                    domain = random.choice(existing_domains)
                    ip = generator.generate_realistic_ip()
                    operations.append({'type': 'add_ip', 'domain': domain, 'ip': ip})
                else:
                    # Se não há domínios, fazer inserção
                    domain = generator.generate_realistic_domain()
                    ip = generator.generate_realistic_ip()
                    operations.append({'type': 'insert', 'domain': domain, 'ip': ip})
                    existing_domains.append(domain)
            
            elif op_type == 'delete':
                if existing_domains:
                    domain = random.choice(existing_domains)
                    operations.append({'type': 'delete', 'domain': domain})
                    existing_domains.remove(domain)  # Remover da lista
                else:
                    # Se não há domínios, fazer busca
                    domain = generator.generate_realistic_domain()
                    operations.append({'type': 'search', 'domain': domain})
        
        return operations
    
    def _choose_operation_type(self, distribution: Dict[str, float]) -> str:
        """Escolhe tipo de operação baseado na distribuição."""
        rand = random.random()
        cumulative = 0.0
        
        for op_type, probability in distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return op_type
        
        # Fallback para o último tipo
        return list(distribution.keys())[-1]
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calcula desvio padrão."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def run_all_benchmarks(self, configs: Optional[List[BenchmarkConfig]] = None) -> List[BenchmarkResult]:
        """
        Executa todos os benchmarks configurados.
        
        Args:
            configs: Lista de configurações (usa padrão se None)
            
        Returns:
            Lista de resultados
        """
        if configs is None:
            configs = self.default_configs
        
        results = []
        tree_types = [TreeType.AVL, TreeType.RED_BLACK]
        
        total_tests = len(configs) * len(tree_types)
        current_test = 0
        
        print(f"Executando {total_tests} testes de benchmark...")
        print("=" * 60)
        
        for config in configs:
            for tree_type in tree_types:
                current_test += 1
                print(f"[{current_test}/{total_tests}] ", end="")
                
                try:
                    result = self.run_benchmark(config, tree_type)
                    results.append(result)
                    
                    print(f"✓ Concluído - Média: {result.mean_total_time:.3f}s "
                          f"({result.mean_operations_per_second:.0f} ops/s)")
                
                except Exception as e:
                    print(f"✗ Erro: {e}")
                
                # Pequena pausa para estabilizar
                time.sleep(0.1)
        
        self.results.extend(results)
        return results
    
    def save_results(self, filename: str, results: Optional[List[BenchmarkResult]] = None) -> None:
        """
        Salva resultados em arquivo JSON.
        
        Args:
            filename: Nome do arquivo
            results: Lista de resultados (usa self.results se None)
        """
        if results is None:
            results = self.results
        
        # Converter resultados para formato serializável
        serializable_results = []
        
        for result in results:
            # Converter execuções individuais
            serializable_runs = []
            for run in result.runs:
                serializable_run = {
                    'total_time': run.total_time,
                    'operations_per_second': run.operations_per_second,
                    'tree_height': run.tree_height,
                    'operation_stats': run.operation_stats,
                    'tree_stats': run.tree_stats
                }
                serializable_runs.append(serializable_run)
            
            serializable_result = {
                'config': {
                    'name': result.config.name,
                    'data_size': result.config.data_size,
                    'operation_count': result.config.operation_count,
                    'benchmark_type': result.config.benchmark_type.value,
                    'operation_distribution': result.config.operation_distribution,
                    'seed': result.config.seed
                },
                'tree_type': result.tree_type.value,
                'runs': serializable_runs,
                'num_runs': result.num_runs,
                'warmup_runs': result.warmup_runs,
                'mean_total_time': result.mean_total_time,
                'std_total_time': result.std_total_time,
                'mean_operations_per_second': result.mean_operations_per_second,
                'std_operations_per_second': result.std_operations_per_second,
                'mean_tree_height': result.mean_tree_height,
                'std_tree_height': result.std_tree_height,
                'aggregated_operation_stats': result.aggregated_operation_stats
            }
            serializable_results.append(serializable_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'benchmark_results': serializable_results,
                'metadata': {
                    'total_tests': len(serializable_results),
                    'timestamp': time.time(),
                    'seed': self.seed,
                    'num_runs': self.num_runs,
                    'warmup_runs': self.warmup_runs
                }
            }, f, indent=2)
        
        print(f"Resultados salvos em {filename}")
    
    def print_summary(self, results: Optional[List[BenchmarkResult]] = None) -> None:
        """Imprime resumo dos resultados."""
        if results is None:
            results = self.results
        
        if not results:
            print("Nenhum resultado disponível.")
            return
        
        print("\n" + "=" * 80)
        print("RESUMO DOS BENCHMARKS")
        print("=" * 80)
        
        # Agrupar por configuração
        configs = {}
        for result in results:
            config_name = result.config.name
            if config_name not in configs:
                configs[config_name] = {}
            configs[config_name][result.tree_type.value] = result
        
        for config_name, tree_results in configs.items():
            print(f"\n{config_name}:")
            print("-" * 40)
            
            for tree_type, result in tree_results.items():
                print(f"  {tree_type}:")
                print(f"    Throughput: {result.mean_operations_per_second:.0f} ± {result.std_operations_per_second:.0f} ops/s")
                print(f"    Tempo total: {result.mean_total_time:.3f} ± {result.std_total_time:.3f} s")
                print(f"    Altura: {result.mean_tree_height:.1f} ± {result.std_tree_height:.1f}")
                print(f"    Execuções: {result.num_runs} (+ {result.warmup_runs} warmup)")
                print()
        
        # Comparação direta para cenário DNS realista
        dns_results = {tree_type: result for tree_type, result in configs.get('DNS_Realistic_1000', {}).items()}
        if len(dns_results) == 2:
            avl_result = dns_results.get('AVL')
            rb_result = dns_results.get('RED_BLACK')
            
            if avl_result and rb_result:
                print(f"\n{'='*60}")
                print("COMPARAÇÃO DIRETA - DNS REALISTA (1000 registros)")
                print("="*60)
                
                throughput_diff = ((rb_result.mean_operations_per_second - avl_result.mean_operations_per_second) 
                                 / avl_result.mean_operations_per_second * 100)
                
                print(f"Throughput:")
                print(f"  AVL: {avl_result.mean_operations_per_second:.0f} ± {avl_result.std_operations_per_second:.0f} ops/s")
                print(f"  Red-Black: {rb_result.mean_operations_per_second:.0f} ± {rb_result.std_operations_per_second:.0f} ops/s")
                print(f"  Diferença: {throughput_diff:+.1f}%")

    def run_comprehensive_benchmarks(self, custom_sizes: List[int]) -> List[BenchmarkResult]:
        """
        Executa suite completa de benchmarks com tamanhos customizados.
        
        Args:
            custom_sizes: Lista de tamanhos de dados customizados
            
        Returns:
            Lista de resultados
        """
        # Criar configurações para todos os tipos de benchmark com tamanhos customizados
        comprehensive_configs = []
        
        # Distribuições de operações para diferentes cenários
        benchmark_scenarios = [
            (BenchmarkType.DNS_REALISTIC, {'search': 0.7, 'insert': 0.15, 'add_ip': 0.1, 'delete': 0.05}),
            (BenchmarkType.INSERT_ONLY, {'search': 0.0, 'insert': 1.0, 'add_ip': 0.0, 'delete': 0.0}),
            (BenchmarkType.SEARCH_ONLY, {'search': 1.0, 'insert': 0.0, 'add_ip': 0.0, 'delete': 0.0}),
            (BenchmarkType.DELETE_ONLY, {'search': 0.0, 'insert': 0.0, 'add_ip': 0.0, 'delete': 1.0}),
        ]
        
        # Criar configurações para cada combinação de tamanho e cenário
        for size in custom_sizes:
            for benchmark_type, distribution in benchmark_scenarios:
                # Calcular número de operações baseado no tamanho
                operation_count = max(100, size // 2)  # Mínimo 100 operações
                
                config = BenchmarkConfig(
                    name=f"{benchmark_type.value.replace('_', ' ').title()}_{size}",
                    data_size=size,
                    operation_count=operation_count,
                    benchmark_type=benchmark_type,
                    operation_distribution=distribution,
                    seed=self.seed
                )
                comprehensive_configs.append(config)
        
        print(f"Configurações criadas: {len(comprehensive_configs)} benchmarks")
        print(f"Tamanhos: {custom_sizes}")
        print(f"Cenários: {[scenario[0].value for scenario in benchmark_scenarios]}")
        print()
        
        # Executar benchmarks com as configurações customizadas
        return self.run_all_benchmarks(comprehensive_configs)


def main():
    """Função principal para execução dos benchmarks."""
    print("=== Sistema de Benchmarks DNS ===\n")
    
    # Criar suite de benchmarks
    benchmark_suite = DNSBenchmarkSuite()
    
    # Executar benchmarks
    results = benchmark_suite.run_all_benchmarks()
    
    # Salvar resultados
    os.makedirs('results', exist_ok=True)
    benchmark_suite.save_results('results/benchmark_results.json')
