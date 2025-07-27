"""
Sistema de Análise e Visualização de Resultados - Versão Refatorada
==================================================================

Este módulo analisa os resultados dos benchmarks e gera visualizações
comparativas entre as estruturas AVL e Rubro-Negra.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from typing import Dict, Any, Tuple
from dataclasses import dataclass

# Configurar estilo seaborn
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.dpi'] = 300

# Configurar fonte DejaVu Sans
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Paleta de cores
COLORS = {
    'AVL': '#2E86AB',      # Azul
    'RED_BLACK': '#A23B72', # Roxo
    'dns_realistic': '#F18F01', # Laranja
    'insert_only': '#C73E1D',   # Vermelho
    'search_only': '#3A86FF',   # Azul claro
    'delete_only': '#8338EC'    # Roxo claro
}
# Paleta para nomes legíveis
COLORS_READABLE = {
    'Árvore AVL': '#2E86AB',
    'Árvore Red-Black': '#A23B72'
}

@dataclass
class PlotConfig:
    """Configuração para gráficos."""
    title: str
    xlabel: str
    ylabel: str
    figsize: Tuple[int, int] = (14, 10)
    palette: str = "husl"
    style: str = "whitegrid"
    context: str = "talk"

class BenchmarkDataLoader:
    """Carregador de dados de benchmark com melhor tratamento de erros."""
    
    def __init__(self, results_file: str):
        """
        Inicializa o carregador de dados.
        
        Args:
            results_file: Caminho para o arquivo de resultados
        """
        self.results_file = results_file
        self.df = None
    
    def _load_results(self) -> Dict[str, Any]:
        """Carrega resultados do arquivo JSON."""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {self.results_file}")
            return {}
        except json.JSONDecodeError:
            print(f"Erro ao decodificar JSON: {self.results_file}")
            return {}
    
    def create_dataframe(self) -> pd.DataFrame:
        """Cria DataFrame limpo e organizado."""
        results = self._load_results()
        if not results:
            return pd.DataFrame()
        
        data_rows = []
        
        # Verificar se é o formato novo (com benchmark_results)
        if 'benchmark_results' in results:
            benchmark_results = results['benchmark_results']
        else:
            # Formato antigo
            benchmark_results = results
        
        for result in benchmark_results:
            config = result.get('config', {})
            tree_type = result.get('tree_type', 'unknown')
            runs = result.get('runs', [])
            
            # Processar cada execução
            for run in runs:
                row = {
                    'benchmark_type': config.get('benchmark_type', 'unknown'),
                    'tree_type': tree_type,
                    'data_size': config.get('data_size', 0),
                    'num_runs': len(runs),
                    'warmup_runs': 0,
                    'total_time': run.get('total_time', 0),
                    'total_time_std': 0.0,
                    'operations_per_second': run.get('operations_per_second', 0),
                    'operations_per_second_std': 0.0,
                    'tree_height': run.get('tree_height', 0),
                    'tree_height_std': 0.0
                }
                
                # Extrair estatísticas de operações
                operation_stats = run.get('operation_stats', {})
                for op_type, stats in operation_stats.items():
                    row.update({
                        f'{op_type}_avg_time': stats.get('avg_time', 0),
                        f'{op_type}_avg_time_std': stats.get('std_dev', 0.0),
                        f'{op_type}_count': stats.get('count', 0),
                        f'{op_type}_total_time': stats.get('total_time', 0)
                    })
                # Extrair rotações
                tree_stats = run.get('tree_stats', {})
                # Corrigir para pegar do campo aninhado se existir
                if 'tree_stats' in tree_stats:
                    stats = tree_stats['tree_stats']
                else:
                    stats = tree_stats
                row['left_rotations'] = stats.get('left_rotations', 0)
                row['right_rotations'] = stats.get('right_rotations', 0)
                row['total_rotations'] = stats.get('total_rotations', 0)
                
                data_rows.append(row)
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            
            # Limpar dados e adicionar colunas úteis
            df = self._clean_and_enhance_dataframe(df)
            return df
        else:
            return pd.DataFrame()
    
    def _extract_base_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai informações básicas do resultado."""
        config = result.get('config', {})
        return {
            'benchmark_type': config.get('benchmark_type', 'unknown'),
            'tree_type': config.get('tree_type', 'unknown'),
            'data_size': config.get('data_size', 0),
            'num_runs': config.get('num_runs', 1),
            'warmup_runs': config.get('warmup_runs', 0)
        }
    
    def _is_new_format(self, result: Dict[str, Any]) -> bool:
        """Verifica se é formato novo (com estatísticas agregadas)."""
        return 'mean_total_time' in result
    
    def _process_new_format(self, base_row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Processa formato novo com estatísticas agregadas."""
        row = base_row.copy()
        row.update({
            'total_time': result['mean_total_time'],
            'total_time_std': result['std_total_time'],
            'operations_per_second': result['mean_operations_per_second'],
            'operations_per_second_std': result['std_operations_per_second'],
            'tree_height': result['mean_tree_height'],
            'tree_height_std': result['std_tree_height'],
            'num_runs': result['num_runs'],
            'warmup_runs': result['warmup_runs']
        })
        
        # Extrair estatísticas de operações agregadas
        if 'aggregated_operation_stats' in result:
            for op_type, stats in result['aggregated_operation_stats'].items():
                row.update({
                    f'{op_type}_avg_time': stats['mean_avg_time'],
                    f'{op_type}_avg_time_std': stats['std_avg_time'],
                    f'{op_type}_count': stats['mean_count'],
                    f'{op_type}_total_time': stats['mean_total_time']
                })
        
        return row
    
    def _process_old_format(self, base_row: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Processa formato antigo (compatibilidade)."""
        row = base_row.copy()
        row.update({
            'total_time': result['total_time'],
            'total_time_std': 0.0,
            'operations_per_second': result['operations_per_second'],
            'operations_per_second_std': 0.0,
            'tree_height': result['tree_height'],
            'tree_height_std': 0.0,
            'num_runs': 1,
            'warmup_runs': 0
        })
        
        # Extrair estatísticas de operações
        if 'operation_stats' in result:
            for op_type, stats in result['operation_stats'].items():
                row.update({
                    f'{op_type}_avg_time': stats['avg_time'],
                    f'{op_type}_avg_time_std': stats.get('std_dev', 0.0),
                    f'{op_type}_count': stats['count'],
                    f'{op_type}_total_time': stats['total_time']
                })
        
        return row
    
    def _clean_and_enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e melhora o DataFrame."""
        # Converter tipos de dados
        df['data_size'] = pd.to_numeric(df['data_size'], errors='coerce')
        df['operations_per_second'] = pd.to_numeric(df['operations_per_second'], errors='coerce')
        df['total_time'] = pd.to_numeric(df['total_time'], errors='coerce')
        df['tree_height'] = pd.to_numeric(df['tree_height'], errors='coerce')
        
        # Adicionar colunas úteis
        df['benchmark_type_readable'] = df['benchmark_type'].map({
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        })
        
        df['tree_type_readable'] = df['tree_type'].map({
            'AVL': 'Árvore AVL',
            'RED_BLACK': 'Árvore Red-Black'
        })
        
        # Calcular eficiência (operações por segundo por elemento)
        df['efficiency'] = df['operations_per_second'] / df['data_size']
        
        return df


class DidacticPlotter:
    """Criador de gráficos acessíveis usando seaborn."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o plotador.
        
        Args:
            df: DataFrame com dados dos benchmarks
        """
        self.df = df
        self.setup_style()
    
    def setup_style(self):
        """Configura estilo visual consistente."""
        sns.set_theme(style="whitegrid", palette="husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    
    def create_benchmark_specific_analysis(self, output_dir: str) -> None:
        """Cria análise específica para cada tipo de benchmark."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Obter tipos únicos de benchmark
        benchmark_types = self.df['benchmark_type'].unique()
        
        print(f"Gerando gráficos para {len(benchmark_types)} tipos de benchmark...")
        
        for bench_type in benchmark_types:
            print(f"  Processando: {bench_type}")
            
            # Filtrar dados para este tipo de benchmark
            bench_data = self.df[self.df['benchmark_type'] == bench_type].copy()
            
            if not bench_data.empty:
                # Criar pasta específica para este benchmark
                bench_dir = os.path.join(output_dir, f'benchmark_{bench_type}')
                os.makedirs(bench_dir, exist_ok=True)
                
                # Gerar gráficos específicos para este benchmark
                self._create_throughput_analysis_for_benchmark(bench_data, bench_type, bench_dir)
                self._create_operation_analysis_for_benchmark(bench_data, bench_type, bench_dir)
                self._create_complexity_analysis_for_benchmark(bench_data, bench_type, bench_dir)
                self._create_comprehensive_dashboard_for_benchmark(bench_data, bench_type, bench_dir)
                # Novo: gráficos de tempos por operação separados por tamanho
                self.create_operation_times_per_size(bench_data, bench_type, bench_dir)
                # Novo: gráficos de rotações
                self.create_rotation_graphs(bench_data, bench_type, bench_dir)
    
    def _create_throughput_analysis_for_benchmark(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Cria análise de throughput específica para um tipo de benchmark."""
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Título baseado no tipo de benchmark
        benchmark_names = {
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        }
        bench_name = benchmark_names.get(bench_type, bench_type.replace('_', ' ').title())
        
        fig.suptitle(f'Análise de Throughput - {bench_name}\nComparação de Desempenho entre Estruturas', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Gráfico de barras de throughput
        ax1 = axes[0, 0]
        self._plot_throughput_bars(data, ax1)
        
        # 2. Gráfico de linha de throughput vs tamanho
        ax2 = axes[0, 1]
        self._plot_throughput_lines(data, ax2)
        
        # 3. Gráfico de eficiência (ops/s por elemento)
        ax3 = axes[1, 0]
        self._plot_efficiency(data, ax3)
        
        # 4. Gráfico de diferença percentual
        ax4 = axes[1, 1]
        self._plot_percentage_difference(data, ax4)
        
        # Ajustar layout
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        plt.savefig(f'{output_dir}/throughput_analysis_{bench_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Gráfico de throughput salvo: {output_dir}/throughput_analysis_{bench_type}.png")
    
    def _create_operation_analysis_for_benchmark(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Cria análise de operações específica para um tipo de benchmark."""
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Título baseado no tipo de benchmark
        benchmark_names = {
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        }
        bench_name = benchmark_names.get(bench_type, bench_type.replace('_', ' ').title())
        
        fig.suptitle(f'Análise de Operações - {bench_name}\nTempos e Distribuições', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        operations = ['search', 'insert', 'add_ip', 'delete']
        operation_names = ['Busca', 'Inserção', 'Adição de IP', 'Remoção']
        
        for idx, (op, name) in enumerate(zip(operations, operation_names)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            self._plot_operation_analysis(data, op, name, ax)
        
        # Ajustar layout
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        plt.savefig(f'{output_dir}/operation_analysis_{bench_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Gráfico de operações salvo: {output_dir}/operation_analysis_{bench_type}.png")
    
    def _create_complexity_analysis_for_benchmark(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Cria análise de complexidade específica para um tipo de benchmark."""
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # Título baseado no tipo de benchmark
        benchmark_names = {
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        }
        bench_name = benchmark_names.get(bench_type, bench_type.replace('_', ' ').title())
        fig.suptitle(f'Análise de Complexidade - {bench_name}\nEstrutura e Eficiência das Árvores', 
                    fontsize=18, fontweight='bold', y=0.98)
        # 1. Altura das árvores
        ax1 = axes[0, 0]
        self._plot_tree_height(data, ax1)
        # 2. Tempo por operação
        ax2 = axes[0, 1]
        self._plot_time_per_operation(data, ax2)
        # 3. Comparação teórica
        ax3 = axes[1, 0]
        self._plot_theoretical_comparison(data, ax3)
        # 4. Eficiência de complexidade
        ax4 = axes[1, 1]
        self._plot_complexity_efficiency(data, ax4)
        # Remover legendas dos subplots
        for ax in axes.flat:
            ax.legend_.remove() if ax.get_legend() else None
        # Adicionar legenda geral fora dos subplots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, title='Tipo de Árvore', loc='center left', bbox_to_anchor=(0.74, 0.5))
        # Ajustar layout para dar espaço à direita
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3, right=0.72)
        plt.savefig(f'{output_dir}/complexity_analysis_{bench_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"    Gráfico de complexidade salvo: {output_dir}/complexity_analysis_{bench_type}.png")
    
    def _create_comprehensive_dashboard_for_benchmark(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Cria dashboard completo específico para um tipo de benchmark."""
        # Criar figura grande
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.5)
        
        # Título baseado no tipo de benchmark
        benchmark_names = {
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        }
        bench_name = benchmark_names.get(bench_type, bench_type.replace('_', ' ').title())
        
        fig.suptitle(f'Dashboard Completo - {bench_name}\nAnálise de Desempenho entre Estruturas', 
                    fontsize=20, fontweight='bold', y=0.92)
        
        # 1. Throughput geral
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_throughput_bars(data, ax1)
        
        # 2. Evolução do throughput
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_throughput_lines(data, ax2)
        
        # 3. Análise de operações
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_operations_summary(data, ax3)
        
        # 4. Altura das árvores
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_tree_height(data, ax4)
        
        # 5. Eficiência
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_efficiency(data, ax5)
        
        # 6. Diferença percentual
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_percentage_difference(data, ax6)
        
        # Ajustar layout
        fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, hspace=0.5, wspace=0.5)
        plt.savefig(f'{output_dir}/comprehensive_dashboard_{bench_type}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"    Dashboard completo salvo: {output_dir}/comprehensive_dashboard_{bench_type}.png")

    def create_throughput_comparison(self, output_dir: str) -> None:
        """Cria gráfico de comparação de throughput."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtrar dados DNS realista
        dns_data = self.df[self.df['benchmark_type'] == 'dns_realistic'].copy()
        
        if dns_data.empty:
            print("Nenhum dado DNS realista encontrado para throughput")
            return
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Throughput\nComparação de Desempenho entre Estruturas', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Gráfico de barras de throughput
        ax1 = axes[0, 0]
        self._plot_throughput_bars(dns_data, ax1)
        
        # 2. Gráfico de linha de throughput vs tamanho
        ax2 = axes[0, 1]
        self._plot_throughput_lines(dns_data, ax2)
        
        # 3. Gráfico de eficiência (ops/s por elemento)
        ax3 = axes[1, 0]
        self._plot_efficiency(dns_data, ax3)
        
        # 4. Gráfico de diferença percentual
        ax4 = axes[1, 1]
        self._plot_percentage_difference(dns_data, ax4)
        
        # Ajustar layout para dar mais espaço ao título
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        plt.savefig(f'{output_dir}/throughput_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Gráfico de throughput salvo: {output_dir}/throughput_analysis.png")
    
    def _plot_throughput_bars(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota gráfico de barras de throughput."""
        # Preparar dados para seaborn
        plot_data = data.groupby(['data_size', 'tree_type_readable'])['operations_per_second'].mean().reset_index()
        
        # Criar gráfico de barras
        sns.barplot(data=plot_data, x='data_size', y='operations_per_second', 
                   hue='tree_type_readable', ax=ax, palette=['#2E86AB', '#A23B72'])
        
        ax.set_title('Throughput por Tamanho dos Dados', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Operações por Segundo')
        ax.legend(title='Tipo de Árvore', loc='lower left', bbox_to_anchor=(0.0, 0.0))
        
        # Adicionar valores nas barras
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    def _plot_throughput_lines(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota gráfico de linha de throughput."""
        # Criar gráfico de linha
        sns.lineplot(data=data, x='data_size', y='operations_per_second', 
                    hue='tree_type_readable', marker='o', ax=ax, 
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
        
        ax.set_title('Evolução do Throughput', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Operações por Segundo')
        ax.legend(title='Tipo de Árvore', loc='lower left', bbox_to_anchor=(0.0, 0.0))
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota gráfico de eficiência."""
        # Calcular eficiência
        data['efficiency'] = data['operations_per_second'] / data['data_size']
        
        sns.lineplot(data=data, x='data_size', y='efficiency', 
                    hue='tree_type_readable', marker='s', ax=ax,
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
        
        ax.set_title('Eficiência (Ops/s por Elemento)', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Eficiência')
        ax.legend(title='Tipo de Árvore', loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_percentage_difference(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota diferença percentual entre estruturas."""
        # Calcular diferença percentual
        pivot_data = data.groupby(['data_size', 'tree_type'])['operations_per_second'].mean().unstack()
        
        if 'AVL' in pivot_data.columns and 'RED_BLACK' in pivot_data.columns:
            pivot_data['difference_pct'] = ((pivot_data['RED_BLACK'] - pivot_data['AVL']) / 
                                          pivot_data['AVL'] * 100)
            
            colors = ['red' if x < 0 else 'green' for x in pivot_data['difference_pct']]
            
            bars = ax.bar(pivot_data.index, pivot_data['difference_pct'], color=colors, alpha=0.7)
            ax.set_title('Diferença Percentual\n(Red-Black vs AVL)', fontweight='bold')
            ax.set_xlabel('Tamanho dos Dados')
            ax.set_ylabel('Diferença (%)')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    def create_operation_analysis(self, output_dir: str) -> None:
        """Cria análise de operações."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtrar dados DNS realista
        dns_data = self.df[self.df['benchmark_type'] == 'dns_realistic'].copy()
        
        if dns_data.empty:
            print("Nenhum dado DNS realista encontrado para análise de operações")
            return
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Operações\nTempos e Distribuições', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        operations = ['search', 'insert', 'add_ip', 'delete']
        operation_names = ['Busca', 'Inserção', 'Adição de IP', 'Remoção']
        
        for idx, (op, name) in enumerate(zip(operations, operation_names)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            self._plot_operation_analysis(dns_data, op, name, ax)
        
        # Ajustar layout para dar mais espaço ao título
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        plt.savefig(f'{output_dir}/operation_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Análise de operações salva: {output_dir}/operation_analysis.png")
    
    def _plot_operation_analysis(self, data: pd.DataFrame, operation: str, name: str, ax: plt.Axes) -> None:
        """Plota análise de uma operação específica."""
        time_col = f'{operation}_avg_time'
        
        if time_col not in data.columns:
            ax.text(0.5, 0.5, f'Dados de {name}\nnão disponíveis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{name}', fontweight='bold')
            return
        
        # Filtrar dados válidos
        valid_data = data[data[time_col].notna()].copy()
        
        if valid_data.empty:
            ax.text(0.5, 0.5, f'Dados de {name}\nnão disponíveis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title(f'{name}', fontweight='bold')
            return
        
        # Converter para milissegundos para melhor visualização
        valid_data[f'{operation}_time_ms'] = valid_data[time_col] * 1000
        
        # Criar gráfico de linha
        sns.lineplot(data=valid_data, x='data_size', y=f'{operation}_time_ms', 
                    hue='tree_type_readable', marker='o', ax=ax,
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=6)
        
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Tempo (ms)')
        ax.legend(title='Tipo de Árvore', loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def create_complexity_analysis(self, output_dir: str) -> None:
        """Cria análise de complexidade."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtrar dados de inserção para análise de complexidade
        insert_data = self.df[self.df['benchmark_type'] == 'insert_only'].copy()
        if insert_data.empty:
            insert_data = self.df[self.df['benchmark_type'] == 'dns_realistic'].copy()
        
        if insert_data.empty:
            print("Nenhum dado encontrado para análise de complexidade")
            return
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Complexidade\nComportamento Assintótico', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Altura das árvores
        ax1 = axes[0, 0]
        self._plot_tree_height(insert_data, ax1)
        
        # 2. Tempo por operação
        ax2 = axes[0, 1]
        self._plot_time_per_operation(insert_data, ax2)
        
        # 3. Comparação com limites teóricos
        ax3 = axes[1, 0]
        self._plot_theoretical_comparison(insert_data, ax3)
        
        # 4. Análise de eficiência
        ax4 = axes[1, 1]
        self._plot_complexity_efficiency(insert_data, ax4)
        
        # Ajustar layout para dar mais espaço ao título
        plt.subplots_adjust(top=0.85, hspace=0.4, wspace=0.3)
        plt.savefig(f'{output_dir}/complexity_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Análise de complexidade salva: {output_dir}/complexity_analysis.png")
    
    def _plot_tree_height(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota altura das árvores."""
        sns.lineplot(data=data, x='data_size', y='tree_height', 
                    hue='tree_type_readable', marker='s', ax=ax,
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
        
        # Adicionar linha teórica log2(n)
        sizes = sorted(data['data_size'].unique())
        if len(sizes) > 1:
            theoretical_x = np.linspace(min(sizes), max(sizes), 100)
            theoretical_y = np.log2(theoretical_x)
            ax.plot(theoretical_x, theoretical_y, '--', color='gray', alpha=0.7, 
                   linewidth=2, label='Teórico: log2(n)')
        
        ax.set_title('Altura das Árvores', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Altura')
        ax.legend(title='Tipo de Árvore', loc='lower right', bbox_to_anchor=(1.0, 0.0))
        ax.grid(True, alpha=0.3)
    
    def _plot_time_per_operation(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota tempo por operação."""
        data['time_per_op'] = data['total_time'] / data['data_size']
        
        sns.lineplot(data=data, x='data_size', y='time_per_op', 
                    hue='tree_type_readable', marker='o', ax=ax,
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
        
        ax.set_title('Tempo por Operação', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Tempo por Operação (s)')
        ax.legend(title='Tipo de Árvore', loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_theoretical_comparison(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota tempo total médio vs tamanho dos dados, comparando com curva teórica O(log n)."""
        import numpy as np
        # Calcular médias por tamanho e tipo de árvore
        grouped = data.groupby(['data_size', 'tree_type_readable'])['total_time'].mean().reset_index()
        sizes = sorted(data['data_size'].unique())
        # Plotar linhas reais
        colors = {'Árvore AVL': '#2E86AB', 'Árvore Red-Black': '#A23B72'}
        for tree_type in grouped['tree_type_readable'].unique():
            y = [grouped[(grouped['data_size'] == n) & (grouped['tree_type_readable'] == tree_type)]['total_time'].values[0]
                 if not grouped[(grouped['data_size'] == n) & (grouped['tree_type_readable'] == tree_type)].empty else np.nan
                 for n in sizes]
            ax.plot(sizes, y, marker='o', label=tree_type, color=colors.get(tree_type, None), linewidth=3)
        # Curva teórica O(log n) normalizada
        logn = np.log2(sizes)
        # Normalizar para começar no mesmo ponto da menor linha real
        min_real = min([min([grouped[(grouped['data_size'] == n) & (grouped['tree_type_readable'] == t)]['total_time'].values[0]
                             for t in grouped['tree_type_readable'].unique()
                             if not grouped[(grouped['data_size'] == n) & (grouped['tree_type_readable'] == t)].empty])
                        for n in sizes])
        logn_norm = logn / logn[0] * min_real if logn[0] > 0 else logn
        ax.plot(sizes, logn_norm, '--', color='gray', linewidth=2, label='Teórico: O(log n)')
        ax.set_title('Tempo Total Médio vs Tamanho dos Dados\n(com curva O(log n))', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Tempo Total Médio (s)')
        ax.grid(True, alpha=0.3)
        # Legenda fora do gráfico
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), title='Estrutura')
        # Ajustar layout do subplot
        ax.figure.tight_layout(rect=[0, 0, 0.85, 1])
    
    def _plot_complexity_efficiency(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota eficiência de complexidade."""
        # Calcular eficiência baseada na altura
        data['height_efficiency'] = data['data_size'] / data['tree_height']
        
        sns.lineplot(data=data, x='data_size', y='height_efficiency', 
                    hue='tree_type_readable', marker='^', ax=ax,
                    palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
        
        ax.set_title('Eficiência de Altura\n(Elementos por Nível)', fontweight='bold')
        ax.set_xlabel('Tamanho dos Dados')
        ax.set_ylabel('Eficiência')
        ax.legend(title='Tipo de Árvore', loc='lower right', bbox_to_anchor=(1.0, 0.0))
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_dashboard(self, output_dir: str) -> None:
        """Cria dashboard completo."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Filtrar dados DNS realista
        dns_data = self.df[self.df['benchmark_type'] == 'dns_realistic'].copy()
        
        if dns_data.empty:
            print("Nenhum dado DNS realista encontrado para dashboard")
            return
        
        # Criar figura grande
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.5)
        
        fig.suptitle('Dashboard Completo de Análise de Desempenho\nSistema DNS com Árvores Balanceadas', 
                    fontsize=20, fontweight='bold', y=0.92)
        
        # 1. Throughput geral
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_throughput_bars(dns_data, ax1)
        
        # 2. Evolução do throughput
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_throughput_lines(dns_data, ax2)
        
        # 3. Análise de operações
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_operations_summary(dns_data, ax3)
        
        # 4. Altura das árvores
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_tree_height(dns_data, ax4)
        
        # 5. Eficiência
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_efficiency(dns_data, ax5)
        
        # 6. Diferença percentual
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_percentage_difference(dns_data, ax6)
        
        # Ajustar layout manualmente para evitar warnings e dar espaço para legendas
        fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1, hspace=0.5, wspace=0.5)
        plt.savefig(f'{output_dir}/comprehensive_dashboard.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✅ Dashboard completo salvo: {output_dir}/comprehensive_dashboard.png")
    
    def _plot_operations_summary(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """Plota resumo das operações separando AVL e Red-Black."""
        operations = ['search', 'insert', 'add_ip', 'delete']
        operation_names = ['Busca', 'Inserção', 'Adição de IP', 'Remoção']
        
        # Calcular tempos médios por operação e tipo de árvore
        op_times = []
        for op, name in zip(operations, operation_names):
            time_col = f'{op}_avg_time'
            if time_col in data.columns:
                # Separar por tipo de árvore
                for tree_type in ['AVL', 'RED_BLACK']:
                    tree_data = data[data['tree_type'] == tree_type]
                    if not tree_data.empty:
                        avg_time = tree_data[time_col].mean() * 1000  # Converter para ms
                        op_times.append({
                            'Operação': name,
                            'Tipo de Árvore': tree_type,
                            'Tempo Médio (ms)': avg_time
                        })
        
        if op_times:
            op_df = pd.DataFrame(op_times)
            
            # Criar gráfico de barras agrupadas
            x = np.arange(len(operation_names))
            width = 0.35
            
            avl_data = op_df[op_df['Tipo de Árvore'] == 'AVL']
            rb_data = op_df[op_df['Tipo de Árvore'] == 'RED_BLACK']
            
            avl_times = [avl_data[avl_data['Operação'] == name]['Tempo Médio (ms)'].iloc[0] 
                         if len(avl_data[avl_data['Operação'] == name]) > 0 else 0 
                         for name in operation_names]
            
            rb_times = [rb_data[rb_data['Operação'] == name]['Tempo Médio (ms)'].iloc[0] 
                       if len(rb_data[rb_data['Operação'] == name]) > 0 else 0 
                       for name in operation_names]
            
            bars1 = ax.bar(x - width/2, avl_times, width, label='AVL', color='#2E86AB', alpha=0.8)
            bars2 = ax.bar(x + width/2, rb_times, width, label='Red-Black', color='#A23B72', alpha=0.8)
            
            ax.set_title('Tempos Médios por Operação', fontweight='bold')
            ax.set_xlabel('Operação')
            ax.set_ylabel('Tempo (ms)')
            ax.set_xticks(x)
            ax.set_xticklabels(operation_names, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Adicionar valores nas barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}ms', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'Dados de operações\nnão disponíveis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.set_title('Tempos por Operação', fontweight='bold')

    def create_operation_times_per_size(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Gera gráficos de barras para cada tamanho de dados e um gráfico de evolução do tempo médio por operação (4 subplots)."""
        operation_labels = ['Busca', 'Inserção', 'Adição de IP', 'Remoção']
        operation_keys = ['search_avg_time', 'insert_avg_time', 'add_ip_avg_time', 'delete_avg_time']
        data_sizes = sorted(data['data_size'].unique())
        benchmark_names = {
            'dns_realistic': 'DNS Realista',
            'insert_only': 'Apenas Inserção',
            'search_only': 'Apenas Busca',
            'delete_only': 'Apenas Remoção'
        }
        bench_name = benchmark_names.get(bench_type, bench_type.replace('_', ' ').title())
        # Criar subpasta para os gráficos
        times_dir = os.path.join(output_dir, 'operation_times_by_size')
        os.makedirs(times_dir, exist_ok=True)
        # Gráficos de barras por tamanho
        for size in data_sizes:
            size_data = data[data['data_size'] == size]
            if size_data.empty:
                continue
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            x = np.arange(len(operation_labels))
            # Dados para cada árvore
            avl_data = size_data[size_data['tree_type'] == 'AVL']
            rb_data = size_data[size_data['tree_type'] == 'RED_BLACK']
            avl_means = [avl_data[key].mean() * 1000 if not avl_data.empty else 0 for key in operation_keys]
            rb_means = [rb_data[key].mean() * 1000 if not rb_data.empty else 0 for key in operation_keys]
            bars1 = plt.bar(x - bar_width/2, avl_means, bar_width, label='AVL', color='#2E86AB')
            bars2 = plt.bar(x + bar_width/2, rb_means, bar_width, label='Red-Black', color='#A23B72')
            plt.xticks(x, operation_labels)
            plt.title(f'Tempos Médios por Operação\n{bench_name} - {size} registros', fontsize=15, fontweight='bold')
            plt.xlabel('Operação')
            plt.ylabel('Tempo Médio (ms)')
            # Adicionar valores em cima das barras
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            # Legenda fora do gráfico
            plt.legend(title='Tipo de Árvore', loc='upper left', bbox_to_anchor=(1.01, 1.0))
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            filename = f'operation_times_{bench_type}_{size}.png'
            plt.savefig(os.path.join(times_dir, filename), dpi=300)
            plt.close()
            print(f"    Gráfico de tempos por operação salvo: {os.path.join(times_dir, filename)}")
        # Gráfico de evolução do tempo médio por operação (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        colors = {'AVL': '#2E86AB', 'RED_BLACK': '#A23B72'}
        for idx, (op_key, op_label) in enumerate(zip(operation_keys, operation_labels)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            for tree_type in ['AVL', 'RED_BLACK']:
                tree_data = data[data['tree_type'] == tree_type]
                means = []
                for size in data_sizes:
                    size_data = tree_data[tree_data['data_size'] == size]
                    mean = size_data[op_key].mean() * 1000 if not size_data.empty else np.nan
                    means.append(mean)
                ax.plot(data_sizes, means, marker='o', label=tree_type, color=colors[tree_type], linewidth=3)
            ax.set_title(f'{op_label}', fontweight='bold')
            ax.set_xlabel('Tamanho dos Dados')
            ax.set_ylabel('Tempo Médio (ms)')
            ax.grid(True, alpha=0.3)
        # Legenda fora do gráfico (apenas uma vez)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Tipo de Árvore', loc='upper left', bbox_to_anchor=(0.92, 0.98))
        fig.suptitle(f'Evolução do Tempo Médio por Operação\n{bench_name}', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.88, 0.95])
        filename = f'evolution_operation_times_{bench_type}.png'
        plt.savefig(os.path.join(times_dir, filename), dpi=300)
        plt.close()
        print(f"    Gráfico de evolução dos tempos salvo: {os.path.join(times_dir, filename)}")

    def create_rotation_graphs(self, data: pd.DataFrame, bench_type: str, output_dir: str) -> None:
        """Gera gráficos de rotações (total, esquerda, direita) por estrutura e tamanho de dados."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        rot_dir = os.path.join(output_dir, 'rotations')
        os.makedirs(rot_dir, exist_ok=True)
        # Gráfico de barras: rotações totais por data_size e árvore
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=data,
            x='data_size',
            y='total_rotations',
            hue='tree_type_readable',
            palette=COLORS_READABLE
        )
        plt.title(f'Rotações Totais por Tamanho de Dados - {bench_type}')
        plt.xlabel('Tamanho dos Dados')
        plt.ylabel('Total de Rotações')
        plt.legend(title='Estrutura', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(rot_dir, f'rotations_total_{bench_type}.png'))
        plt.close()
        # Gráfico de linhas: evolução das rotações totais
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=data,
            x='data_size',
            y='total_rotations',
            hue='tree_type_readable',
            marker='o',
            linewidth=3,
            palette=COLORS_READABLE
        )
        plt.title(f'Evolução das Rotações Totais - {bench_type}')
        plt.xlabel('Tamanho dos Dados')
        plt.ylabel('Total de Rotações')
        plt.legend(title='Estrutura', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(rot_dir, f'rotations_total_evolution_{bench_type}.png'))
        plt.close()
        # Gráfico de barras: rotações esquerda/direita
        for rot_type, label in [('left_rotations', 'Rotações à Esquerda'), ('right_rotations', 'Rotações à Direita')]:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                data=data,
                x='data_size',
                y=rot_type,
                hue='tree_type_readable',
                palette=COLORS_READABLE
            )
            plt.title(f'{label} por Tamanho de Dados - {bench_type}')
            plt.xlabel('Tamanho dos Dados')
            plt.ylabel(label)
            plt.legend(title='Estrutura', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(rot_dir, f'{rot_type}_{bench_type}.png'))
            plt.close()
        # Gráfico de linhas: evolução esquerda/direita
        for rot_type, label in [('left_rotations', 'Rotações à Esquerda'), ('right_rotations', 'Rotações à Direita')]:
            plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=data,
                x='data_size',
                y=rot_type,
                hue='tree_type_readable',
                marker='o',
                linewidth=3,
                palette=COLORS_READABLE
            )
            plt.title(f'Evolução das {label} - {bench_type}')
            plt.xlabel('Tamanho dos Dados')
            plt.ylabel(label)
            plt.legend(title='Estrutura', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(rot_dir, f'{rot_type}_evolution_{bench_type}.png'))
            plt.close()


class BenchmarkAnalyzer:
    """Analisador de benchmarks com visualizações."""
    
    def __init__(self, results_file: str):
        """
        Inicializa o analisador.
        
        Args:
            results_file: Caminho para o arquivo de resultados
        """
        self.results_file = results_file
        self.data_loader = BenchmarkDataLoader(results_file)
        self.df = self.data_loader.create_dataframe()
        self.plotter = DidacticPlotter(self.df)
    
    def generate_comprehensive_analysis(self, output_dir: str = 'results/analysis') -> None:
        """Gera análise completa com visualizações, apenas gráficos específicos por benchmark."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.df.empty:
            print("Nenhum dado encontrado para análise")
            return
        
        print("Gerando visualizações...")
        
        # Gerar apenas gráficos específicos por benchmark
        print("\n=== Gerando Gráficos Específicos por Benchmark ===")
        self.plotter.create_benchmark_specific_analysis(output_dir)
        
        # Gerar tabelas e gráficos individuais
        self._generate_tables_and_individual_graphs(output_dir)
        
        print("\nAnálise completa gerada com sucesso!")
        print(f"Resultados salvos em: {output_dir}")
        print("\nEstrutura de arquivos gerados:")
        print("  - Gráficos específicos por benchmark: results/analysis/benchmark_*/")
        print("  - Tabelas: results/analysis/tables/")
        print("  - Gráficos individuais: results/analysis/graphs/")
    
    def _generate_tables_and_individual_graphs(self, output_dir: str) -> None:
        """Gera tabelas e gráficos individuais."""
        # Criar pastas
        tables_dir = os.path.join(output_dir, 'tables')
        graphs_dir = os.path.join(output_dir, 'graphs')
        os.makedirs(tables_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Gerar tabelas
        self._generate_throughput_table(tables_dir)
        self._generate_operation_times_table(tables_dir)
        self._generate_general_comparison_table(tables_dir)
        
        # Gerar gráficos individuais
        self._generate_individual_throughput_graph(graphs_dir)
        self._generate_individual_operation_graph(graphs_dir)
        self._generate_individual_tree_height_graph(graphs_dir)
    
    def _generate_throughput_table(self, tables_dir: str) -> None:
        """Gera tabela de throughput."""
        dns_data = self.df[self.df['benchmark_type'] == 'dns_realistic']
        
        if not dns_data.empty:
            # Criar tabela pivot
            pivot_data = dns_data.groupby(['data_size', 'tree_type'])['operations_per_second'].mean().reset_index()
            pivot_table = pivot_data.pivot(index='data_size', columns='tree_type', values='operations_per_second')
            
            # Adicionar diferença percentual
            if 'AVL' in pivot_table.columns and 'RED_BLACK' in pivot_table.columns:
                pivot_table['Diferença (%)'] = ((pivot_table['RED_BLACK'] - pivot_table['AVL']) / 
                                              pivot_table['AVL'] * 100).round(2)
            
            # Salvar CSV
            pivot_table.to_csv(f'{tables_dir}/throughput_table.csv')
            
            # Salvar TXT
            with open(f'{tables_dir}/throughput_table.txt', 'w', encoding='utf-8') as f:
                f.write("TABELA DE THROUGHPUT (Operações por Segundo)\n")
                f.write("=" * 60 + "\n\n")
                f.write(pivot_table.to_string())
                f.write("\n\n")
                f.write("Nota: Valores positivos na coluna 'Diferença (%)' indicam que Red-Black é mais rápida.\n")
    
    def _generate_operation_times_table(self, tables_dir: str) -> None:
        """Gera tabelas de tempos de operação separadas por tipo de benchmark."""
        benchmark_types = self.df['benchmark_type'].unique()
        for bench_type in benchmark_types:
            bench_df = self.df[self.df['benchmark_type'] == bench_type]
        operation_data = []
        for _, row in bench_df.iterrows():
            for op_type in ['search', 'insert', 'add_ip', 'delete']:
                avg_time_col = f'{op_type}_avg_time'
                if avg_time_col in row and pd.notna(row[avg_time_col]):
                    operation_data.append({
                        'Operação': op_type.replace('_', ' ').title(),
                        'Tipo de Árvore': row['tree_type'],
                        'Tamanho dos Dados': row['data_size'],
                        'Tempo Médio (ms)': round(row[avg_time_col] * 1000, 3)
                    })
        if operation_data:
            op_df = pd.DataFrame(operation_data)
            # Criar tabela pivot
            pivot_table = op_df.pivot_table(
                index=['Operação', 'Tamanho dos Dados'], 
                columns='Tipo de Árvore', 
                values='Tempo Médio (ms)',
                aggfunc='mean'
            )
            # Nome amigável para o cenário
            bench_name = bench_type
            # Salvar arquivos
            pivot_table.to_csv(f'{tables_dir}/operation_times_table_{bench_name}.csv')
            with open(f'{tables_dir}/operation_times_table_{bench_name}.txt', 'w', encoding='utf-8') as f:
                f.write(f'TABELA DE TEMPOS DE OPERAÇÃO ({bench_name}) (Milissegundos)\n')
                f.write('=' * 70 + '\n\n')
                f.write(pivot_table.to_string())
                f.write('\n\n')
    
    def _generate_general_comparison_table(self, tables_dir: str) -> None:
        """Gera tabela de comparação geral."""
        summary_data = []
        
        for bench_type in self.df['benchmark_type'].unique():
            type_data = self.df[self.df['benchmark_type'] == bench_type]
            
            for tree_type in ['AVL', 'RED_BLACK']:
                tree_data = type_data[type_data['tree_type'] == tree_type]
                
                if not tree_data.empty:
                    summary_data.append({
                        'Tipo de Benchmark': bench_type.replace('_', ' ').title(),
                        'Tipo de Árvore': tree_type,
                        'Throughput Médio (ops/s)': round(tree_data['operations_per_second'].mean(), 0),
                        'Tempo Médio (s)': round(tree_data['total_time'].mean(), 4),
                        'Altura Média': round(tree_data['tree_height'].mean(), 1)
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Salvar arquivos
            summary_df.to_csv(f'{tables_dir}/general_comparison.csv', index=False)
            
            with open(f'{tables_dir}/general_comparison.txt', 'w', encoding='utf-8') as f:
                f.write("TABELA DE COMPARAÇÃO GERAL\n")
                f.write("=" * 80 + "\n\n")
                f.write(summary_df.to_string(index=False))
    
    def _generate_individual_throughput_graph(self, graphs_dir: str) -> None:
        """Gera gráfico individual de throughput."""
        dns_data = self.df[self.df['benchmark_type'] == 'dns_realistic']
        
        if not dns_data.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Gráfico de linha de throughput
            sns.lineplot(data=dns_data, x='data_size', y='operations_per_second', 
                        hue='tree_type_readable', marker='o', ax=ax,
                        palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
            
            ax.set_title('Throughput vs Tamanho dos Dados', fontweight='bold', fontsize=16)
            ax.set_xlabel('Tamanho dos Dados', fontsize=14)
            ax.set_ylabel('Operações por Segundo', fontsize=14)
            ax.legend(title='Tipo de Árvore', loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{graphs_dir}/throughput_comparison.png', bbox_inches='tight', dpi=300)
            plt.close()
    
    def _generate_individual_operation_graph(self, graphs_dir: str) -> None:
        """Gera gráfico individual de tempos de operação."""
        operations = ['search', 'insert', 'add_ip', 'delete']
        operation_names = ['Busca', 'Inserção', 'Adição de IP', 'Remoção']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Tempos de Operação por Tipo de Árvore', fontsize=16, fontweight='bold')
        
        for idx, (op, name) in enumerate(zip(operations, operation_names)):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            time_col = f'{op}_avg_time'
            if time_col in self.df.columns:
                valid_data = self.df[self.df[time_col].notna()].copy()
                valid_data[f'{op}_time_ms'] = valid_data[time_col] * 1000
                
                sns.lineplot(data=valid_data, x='data_size', y=f'{op}_time_ms', 
                            hue='tree_type_readable', marker='s', ax=ax,
                            palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=6)
                
                ax.set_title(f'{name}', fontweight='bold')
                ax.set_xlabel('Tamanho dos Dados')
                ax.set_ylabel('Tempo (ms)')
                ax.legend(title='Tipo de Árvore', loc='upper left')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Dados de {name}\nnão disponíveis', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{name}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{graphs_dir}/operation_times.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def _generate_individual_tree_height_graph(self, graphs_dir: str) -> None:
        """Gera gráfico individual de altura das árvores."""
        insert_data = self.df[self.df['benchmark_type'] == 'insert_only']
        if insert_data.empty:
            insert_data = self.df[self.df['benchmark_type'] == 'dns_realistic']
        
        if not insert_data.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Gráfico de altura das árvores
            sns.lineplot(data=insert_data, x='data_size', y='tree_height', 
                        hue='tree_type_readable', marker='o', ax=ax,
                        palette=['#2E86AB', '#A23B72'], linewidth=2, markersize=8)
            
            # Adicionar linha teórica log2(n)
            sizes = sorted(insert_data['data_size'].unique())
            if len(sizes) > 1:
                theoretical_x = np.linspace(min(sizes), max(sizes), 100)
                theoretical_y = np.log2(theoretical_x)
                ax.plot(theoretical_x, theoretical_y, '--', color='gray', alpha=0.7, 
                       linewidth=2, label='Teórico: log2(n)')
            
            ax.set_title('Altura das Árvores vs Tamanho dos Dados', fontweight='bold', fontsize=16)
            ax.set_xlabel('Tamanho dos Dados', fontsize=14)
            ax.set_ylabel('Altura', fontsize=14)
            ax.legend(title='Tipo de Árvore', loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{graphs_dir}/tree_height.png', bbox_inches='tight', dpi=300)
            plt.close()
    



def main():
    """Função principal para análise de benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análise de benchmarks DNS')
    parser.add_argument('--results', default='results/benchmark_results.json',
                       help='Arquivo de resultados (padrão: results/benchmark_results.json)')
    parser.add_argument('--output', default='results/analysis',
                       help='Diretório de saída (padrão: results/analysis)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        print(f"Arquivo de resultados não encontrado: {args.results}")
        return
    
    analyzer = BenchmarkAnalyzer(args.results)
    analyzer.generate_comprehensive_analysis(args.output)


if __name__ == "__main__":
    main()

