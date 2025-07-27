#!/usr/bin/env python3
"""
CLI do Sistema DNS Personalizado
================================

Interface de linha de comando para o sistema DNS que permite:
- Executar benchmarks com parâmetros customizáveis
- Analisar resultados de benchmarks
"""

import argparse
import sys
import os
import time
import random

# Adicionar diretórios ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_structures'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'benchmarks'))

from dns_system import DNSSystem, TreeType
from dns_data_generator import DNSDataGenerator
from benchmarks.benchmark_system import DNSBenchmarkSuite, BenchmarkConfig, BenchmarkType
from analysis_visualization import BenchmarkAnalyzer


class DNSCLIError(Exception):
    """Exceção personalizada para erros da CLI."""
    pass


class DNSCLI:
    """Interface de linha de comando para o sistema DNS."""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Cria o parser de argumentos da CLI."""
        parser = argparse.ArgumentParser(
            description='Sistema DNS Personalizado - Comparação AVL vs Red-Black',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Exemplos de uso:

  # Executar benchmark com tamanhos customizados
  python dns_cli.py benchmark --sizes 100,1000,5000 --operations 90,5,3,2

  # Gerar dados DNS realistas
  python dns_cli.py generate --size 2000 --output my_data.json

  # Modo interativo para testar operações
  python dns_cli.py interactive --tree-type red-black

  # Analisar resultados existentes
  python dns_cli.py analyze --input results/benchmark_results.json

  # Executar benchmark completo com configurações padrão
  python dns_cli.py benchmark --full
  
  # Executar suite completa com tamanhos customizados
  python dns_cli.py benchmark --comprehensive --sizes 500,2000,8000
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Comandos disponíveis')
        
        # Comando benchmark
        self._add_benchmark_parser(subparsers)
        
        # Comando generate
        self._add_generate_parser(subparsers)
        
        # Comando interactive
        self._add_interactive_parser(subparsers)
        
        # Comando analyze
        self._add_analyze_parser(subparsers)
        
        return parser
    
    def _add_benchmark_parser(self, subparsers):
        """Adiciona parser para comando benchmark."""
        benchmark_parser = subparsers.add_parser(
            'benchmark',
            help='Executar benchmarks de desempenho'
        )
        
        benchmark_parser.add_argument(
            '--sizes',
            type=str,
            default='100,1000,10000',
            help='Tamanhos de dados para teste (separados por vírgula). Padrão: 100,1000,10000'
        )
        
        benchmark_parser.add_argument(
            '--operations',
            type=str,
            default='90,5,3,2',
            help='Distribuição de operações DNS: busca,inserção,adição_ip,remoção (valores separados por vírgula). Padrão: 90,5,3,2'
        )
        
        benchmark_parser.add_argument(
            '--operation-count',
            type=int,
            help='Número de operações por benchmark (padrão: 2x o tamanho dos dados)'
        )
        
        benchmark_parser.add_argument(
            '--output',
            type=str,
            default='results/benchmark_results.json',
            help='Arquivo de saída para resultados. Padrão: results/benchmark_results.json'
        )
        
        benchmark_parser.add_argument(
            '--full',
            action='store_true',
            help='Executar suite completa de benchmarks (ignora outros parâmetros)'
        )
        
        benchmark_parser.add_argument(
            '--comprehensive',
            action='store_true',
            help='Executar suite completa de benchmarks com tamanhos customizados (usa --sizes especificado)'
        )
        
        benchmark_parser.add_argument(
            '--analyze',
            action='store_true',
            help='Gerar análise e visualizações após benchmarks'
        )
        
        benchmark_parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help='Semente para reprodutibilidade. Padrão: aleatória'
        )
        
        benchmark_parser.add_argument(
            '--num-runs',
            type=int,
            default=5,
            help='Número de execuções por benchmark. Padrão: 5'
        )
        
        benchmark_parser.add_argument(
            '--warmup-runs',
            type=int,
            default=2,
            help='Número de execuções de aquecimento (descartadas). Padrão: 2'
        )
    
    def _add_generate_parser(self, subparsers):
        """Adiciona parser para comando generate."""
        generate_parser = subparsers.add_parser(
            'generate',
            help='Gerar dados DNS realistas'
        )
        
        generate_parser.add_argument(
            '--size',
            type=int,
            required=True,
            help='Número de domínios a gerar'
        )
        
        generate_parser.add_argument(
            '--output',
            type=str,
            required=True,
            help='Arquivo de saída (formato JSON)'
        )
        
        generate_parser.add_argument(
            '--distribution',
            type=str,
            default='40,35,25',
            help='Distribuição de IPs por domínio: 1_ip,2_ips,3_ips (valores separados por vírgula). Padrão: 40,35,25'
        )
        
        generate_parser.add_argument(
            '--seed',
            type=int,
            default=None,
            help='Semente para reprodutibilidade. Padrão: aleatória'
        )
    
    def _add_interactive_parser(self, subparsers):
        """Adiciona parser para comando interactive."""
        interactive_parser = subparsers.add_parser(
            'interactive',
            help='Modo interativo para testar operações DNS'
        )
        
        interactive_parser.add_argument(
            '--tree-type',
            type=str,
            choices=['avl', 'red-black'],
            default='avl',
            help='Tipo de árvore a usar. Padrão: avl'
        )
        
        interactive_parser.add_argument(
            '--load-data',
            type=str,
            help='Arquivo JSON com dados para carregar inicialmente'
        )
    
    def _add_analyze_parser(self, subparsers):
        """Adiciona parser para comando analyze."""
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='Analisar resultados de benchmarks'
        )
        
        analyze_parser.add_argument(
            '--input',
            type=str,
            required=True,
            help='Arquivo JSON com resultados de benchmark'
        )
        
        analyze_parser.add_argument(
            '--output-dir',
            type=str,
            default='results/analysis',
            help='Diretório para salvar análises. Padrão: results/analysis'
        )
    
    def run(self, args=None):
        """Executa a CLI com os argumentos fornecidos."""
        if args is None:
            args = sys.argv[1:]
        
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return
        
        try:
            if parsed_args.command == 'benchmark':
                self._run_benchmark(parsed_args)
            elif parsed_args.command == 'generate':
                self._run_generate(parsed_args)
            elif parsed_args.command == 'interactive':
                self._run_interactive(parsed_args)
            elif parsed_args.command == 'analyze':
                self._run_analyze(parsed_args)
        
        except DNSCLIError as e:
            print(f"Erro: {e}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            sys.exit(1)
        except Exception as e:
            print(f"Erro inesperado: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _run_benchmark(self, args):
        """Executa benchmarks de desempenho."""
        print("=== Sistema DNS - Benchmarks de Desempenho ===\n")
        
        # Gerar seed aleatória se não especificada
        if args.seed is None:
            args.seed = random.randint(1, 999999)
            print(f"Usando seed aleatória: {args.seed}\n")
        
        if args.full:
            print("Executando suite completa de benchmarks...")
            benchmark_suite = DNSBenchmarkSuite(args.seed, args.num_runs, args.warmup_runs)
            results = benchmark_suite.run_all_benchmarks()
            
            # Salvar resultados
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            benchmark_suite.save_results(args.output)
            
            # Mostrar resumo
            benchmark_suite.print_summary()
            
        elif args.comprehensive:
            print("Executando suite completa de benchmarks com tamanhos customizados...")
            
            # Processar tamanhos customizados
            sizes = [int(x.strip()) for x in args.sizes.split(',')]
            print(f"Tamanhos de dados: {sizes}")
            print(f"Execuções por benchmark: {args.num_runs} (+ {args.warmup_runs} warmup)")
            print()
            
            # Criar suite de benchmark com tamanhos customizados
            benchmark_suite = DNSBenchmarkSuite(args.seed, args.num_runs, args.warmup_runs)
            results = benchmark_suite.run_comprehensive_benchmarks(sizes)
            
            # Salvar resultados
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            benchmark_suite.save_results(args.output)
            
            # Mostrar resumo
            benchmark_suite.print_summary()
            
        else:
            # Benchmark customizado
            print("Executando benchmark customizado...")
            sizes = [int(x.strip()) for x in args.sizes.split(',')]
            operations = [int(x.strip()) for x in args.operations.split(',')]
            
            if len(operations) != 4:
                raise DNSCLIError("Distribuição de operações deve ter 4 valores: busca,inserção,adição_ip,remoção")
            
            if abs(sum(operations) - 100) > 0.1:
                raise DNSCLIError("Distribuição de operações deve somar 100%")
            
            # Converter para proporções
            operations = [x / 100.0 for x in operations]
            operation_distribution = {
                'search': operations[0],
                'insert': operations[1],
                'add_ip': operations[2],
                'delete': operations[3]
            }
            
            print(f"Tamanhos de dados: {sizes}")
            print(f"Distribuição de operações: {operation_distribution}")
            print(f"Execuções por benchmark: {args.num_runs} (+ {args.warmup_runs} warmup)")
            print()
            
            # Criar configurações customizadas
            configs = []
            for size in sizes:
                operation_count = args.operation_count or size * 2
                
                config = BenchmarkConfig(
                    name=f"Custom_DNS_{size}",
                    data_size=size,
                    operation_count=operation_count,
                    benchmark_type=BenchmarkType.DNS_REALISTIC,
                    operation_distribution=operation_distribution,
                    seed=args.seed
                )
                configs.append(config)
            
            # Executar benchmarks
            benchmark_suite = DNSBenchmarkSuite(args.seed, args.num_runs, args.warmup_runs)
            results = benchmark_suite.run_all_benchmarks(configs)
            
            # Salvar resultados
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            benchmark_suite.save_results(args.output)
            
            # Mostrar resumo
            benchmark_suite.print_summary()
        
        # Gerar análise se solicitado
        if args.analyze:
            self._run_analyze(argparse.Namespace(input=args.output, output_dir='results/analysis'))
    
    def _run_generate(self, args):
        """Gera dados DNS realistas."""
        print("=== Gerador de Dados DNS ===\n")
        
        # Gerar seed aleatória se não especificada
        if args.seed is None:
            args.seed = random.randint(1, 999999)
            print(f"Usando seed aleatória: {args.seed}")
            print()
        
        distribution = [int(x.strip()) for x in args.distribution.split(',')]
        if len(distribution) != 3:
            raise DNSCLIError("Distribuição deve ter 3 valores: 1_ip,2_ips,3_ips")
        
        if abs(sum(distribution) - 100) > 0.1:
            raise DNSCLIError("Distribuição deve somar 100%")
        
        print(f"Gerando {args.size} domínios...")
        print(f"Distribuição de IPs: {distribution[0]}% (1 IP), {distribution[1]}% (2 IPs), {distribution[2]}% (3 IPs)")
        
        # Criar gerador
        generator = DNSDataGenerator(args.seed)
        
        # Gerar dados
        start_time = time.time()
        domain_ip_pairs = generator.generate_domain_ip_pairs(args.size)
        end_time = time.time()
        
        # Salvar dados
        output_dir = os.path.dirname(args.output)
        if output_dir:  # Só criar se não for diretório atual
            os.makedirs(output_dir, exist_ok=True)
        generator.save_to_file(domain_ip_pairs, args.output)
        
        # Mostrar estatísticas
        distribution_stats = generator._calculate_distribution(domain_ip_pairs)
        
        print(f"\nDados gerados em {end_time - start_time:.2f} segundos")
        print(f"Arquivo salvo: {args.output}")
        print("\nDistribuição real de IPs por domínio:")
        for key, value in distribution_stats.items():
            print(f"  {key}: {value:.1f}%")
    
    def _run_interactive(self, args):
        """Executa modo interativo."""
        print("=== Modo Interativo do Sistema DNS ===\n")
        
        # Criar sistema DNS
        tree_type = TreeType.AVL if args.tree_type == 'avl' else TreeType.RED_BLACK
        dns_system = DNSSystem(tree_type)
        
        print(f"Usando árvore: {tree_type.value}")
        
        # Carregar dados se especificado
        if args.load_data:
            if not os.path.exists(args.load_data):
                raise DNSCLIError(f"Arquivo não encontrado: {args.load_data}")
            
            print(f"Carregando dados de: {args.load_data}")
            loaded_count = dns_system.load_data_from_file(args.load_data)
            print(f"Carregados {loaded_count} domínios")
        
        print("\nComandos disponíveis:")
        print("  add <domínio> <ip>        - Adicionar domínio")
        print("  search <domínio>          - Buscar domínio")
        print("  remove <domínio>          - Remover domínio")
        print("  addip <domínio> <ip>      - Adicionar IP a domínio")
        print("  removeip <domínio> <ip>   - Remover IP de domínio")
        print("  list                      - Listar todos os domínios")
        print("  stats                     - Mostrar estatísticas")
        print("  help                      - Mostrar esta ajuda")
        print("  quit                      - Sair\n")
        
        while True:
            try:
                command = input("DNS> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd == 'quit' or cmd == 'exit':
                    break
                
                elif cmd == 'help':
                    print("Comandos disponíveis:")
                    print("  add <domínio> <ip>        - Adicionar domínio")
                    print("  search <domínio>          - Buscar domínio")
                    print("  remove <domínio>          - Remover domínio")
                    print("  addip <domínio> <ip>      - Adicionar IP a domínio")
                    print("  removeip <domínio> <ip>   - Remover IP de domínio")
                    print("  list                      - Listar todos os domínios")
                    print("  stats                     - Mostrar estatísticas")
                    print("  help                      - Mostrar esta ajuda")
                    print("  quit                      - Sair")
                
                elif cmd == 'add':
                    if len(command) != 3:
                        print("Uso: add <domínio> <ip>")
                        continue
                    
                    domain, ip = command[1], command[2]
                    success = dns_system.add_domain(domain, ip)
                    if success:
                        print(f"✓ Domínio {domain} adicionado com IP {ip}")
                    else:
                        print(f"✗ Domínio {domain} já existe")
                
                elif cmd == 'search':
                    if len(command) != 2:
                        print("Uso: search <domínio>")
                        continue
                    
                    domain = command[1]
                    ips = dns_system.search_domain(domain)
                    if ips:
                        print(f"✓ {domain} -> {', '.join(ips)}")
                    else:
                        print(f"✗ Domínio {domain} não encontrado")
                
                elif cmd == 'remove':
                    if len(command) != 2:
                        print("Uso: remove <domínio>")
                        continue
                    
                    domain = command[1]
                    success = dns_system.remove_domain(domain)
                    if success:
                        print(f"✓ Domínio {domain} removido")
                    else:
                        print(f"✗ Domínio {domain} não encontrado")
                
                elif cmd == 'addip':
                    if len(command) != 3:
                        print("Uso: addip <domínio> <ip>")
                        continue
                    
                    domain, ip = command[1], command[2]
                    success = dns_system.add_ip_to_domain(domain, ip)
                    if success:
                        print(f"✓ IP {ip} adicionado ao domínio {domain}")
                    else:
                        print(f"✗ Falha ao adicionar IP (domínio não existe ou IP já existe)")
                
                elif cmd == 'removeip':
                    if len(command) != 3:
                        print("Uso: removeip <domínio> <ip>")
                        continue
                    
                    domain, ip = command[1], command[2]
                    success = dns_system.remove_ip_from_domain(domain, ip)
                    if success:
                        print(f"✓ IP {ip} removido do domínio {domain}")
                    else:
                        print(f"✗ Falha ao remover IP (domínio ou IP não existe)")
                
                elif cmd == 'list':
                    domains = dns_system.list_all_domains()
                    if domains:
                        print(f"Domínios cadastrados ({len(domains)} total):")
                        for domain, ips in domains:
                            print(f"  {domain} -> {', '.join(ips)}")
                    else:
                        print("Nenhum domínio cadastrado")
                
                elif cmd == 'stats':
                    dns_system.print_system_info()
                
                else:
                    print(f"Comando desconhecido: {cmd}. Digite 'help' para ver comandos disponíveis.")
            
            except EOFError:
                break
            except Exception as e:
                print(f"Erro: {e}")
        
        print("\nSaindo do modo interativo...")
    
    def _run_analyze(self, args):
        """Analisa resultados de benchmarks."""
        print("=== Análise de Resultados de Benchmark ===\n")
        
        if not os.path.exists(args.input):
            raise DNSCLIError(f"Arquivo não encontrado: {args.input}")
        
        print(f"Analisando resultados de: {args.input}")
        print(f"Salvando análise em: {args.output_dir}")
        
        # Criar analisador
        analyzer = BenchmarkAnalyzer(args.input)
        
        # Gerar análise completa
        analyzer.generate_comprehensive_analysis(args.output_dir)
        
        print("\nAnálise concluída!")


def main():
    """Função principal da CLI."""
    cli = DNSCLI()
    cli.run()


if __name__ == "__main__":
    main()

