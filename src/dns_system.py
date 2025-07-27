"""
Sistema DNS Personalizado
========================

Este módulo implementa um sistema DNS que utiliza árvores AVL e Rubro-Negra
para armazenar e gerenciar registros DNS. Fornece uma interface unificada
para operações básicas de DNS.

Operações suportadas:
- Inserção de domínios
- Busca de domínios
- Remoção de domínios
- Adição de IPs a domínios existentes
- Remoção de IPs de domínios
- Listagem de todos os registros
"""

import sys
import os
import time
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

# Adicionar o diretório data_structures ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_structures'))

from data_structures.avl_tree import AVLTree
from data_structures.red_black_tree import RedBlackTree
from dns_data_generator import DNSDataGenerator

class TreeType(Enum):
    """Tipos de árvore disponíveis."""
    AVL = "AVL"
    RED_BLACK = "RED_BLACK"

class DNSSystem:
    """
    Sistema DNS que utiliza árvores balanceadas para armazenamento eficiente.
    
    Suporta tanto árvores AVL quanto Rubro-Negra, permitindo comparação
    de desempenho entre as duas estruturas.
    """
    
    def __init__(self, tree_type: TreeType = TreeType.AVL):
        """
        Inicializa o sistema DNS com o tipo de árvore especificado.
        
        Args:
            tree_type: Tipo de árvore a ser utilizada (AVL ou RED_BLACK)
        """
        self.tree_type = tree_type
        
        if tree_type == TreeType.AVL:
            self.tree = AVLTree()
        else:
            self.tree = RedBlackTree()
        
        # Estatísticas do sistema
        self.total_operations = 0
        self.operation_history = []
    
    def add_domain(self, domain: str, ip: str) -> bool:
        """
        Adiciona um novo domínio com IP ao sistema.
        
        Args:
            domain: Nome do domínio
            ip: Endereço IP associado
            
        Returns:
            True se adicionado com sucesso, False se domínio já existe
        """
        start_time = time.perf_counter()
        result = self.tree.insert(domain, ip)
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'add_domain',
            'domain': domain,
            'ip': ip,
            'time': operation_time,
            'success': result
        })
        
        return result
    
    def search_domain(self, domain: str) -> Optional[List[str]]:
        """
        Busca um domínio no sistema.
        
        Args:
            domain: Nome do domínio a buscar
            
        Returns:
            Lista de IPs associados ao domínio ou None se não encontrado
        """
        start_time = time.perf_counter()
        result = self.tree.search(domain)
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'search_domain',
            'domain': domain,
            'time': operation_time,
            'success': result is not None,
            'result_count': len(result) if result else 0
        })
        
        return result
    
    def remove_domain(self, domain: str) -> bool:
        """
        Remove um domínio do sistema.
        
        Args:
            domain: Nome do domínio a remover
            
        Returns:
            True se removido com sucesso, False se domínio não existe
        """
        start_time = time.perf_counter()
        result = self.tree.delete(domain)
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'remove_domain',
            'domain': domain,
            'time': operation_time,
            'success': result
        })
        
        return result
    
    def add_ip_to_domain(self, domain: str, ip: str) -> bool:
        """
        Adiciona um IP a um domínio existente.
        
        Args:
            domain: Nome do domínio
            ip: Endereço IP a adicionar
            
        Returns:
            True se adicionado com sucesso, False se domínio não existe ou IP já existe
        """
        start_time = time.perf_counter()
        result = self.tree.add_ip_to_domain(domain, ip)
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'add_ip_to_domain',
            'domain': domain,
            'ip': ip,
            'time': operation_time,
            'success': result
        })
        
        return result
    
    def remove_ip_from_domain(self, domain: str, ip: str) -> bool:
        """
        Remove um IP de um domínio.
        
        Args:
            domain: Nome do domínio
            ip: Endereço IP a remover
            
        Returns:
            True se removido com sucesso, False se domínio ou IP não existe
        """
        start_time = time.perf_counter()
        result = self.tree.remove_ip_from_domain(domain, ip)
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'remove_ip_from_domain',
            'domain': domain,
            'ip': ip,
            'time': operation_time,
            'success': result
        })
        
        return result
    
    def list_all_domains(self) -> List[Tuple[str, List[str]]]:
        """
        Lista todos os domínios e seus IPs.
        
        Returns:
            Lista de tuplas (domínio, lista_de_ips)
        """
        start_time = time.perf_counter()
        result = self.tree.get_all_domains()
        end_time = time.perf_counter()
        
        operation_time = end_time - start_time
        self.total_operations += 1
        self.operation_history.append({
            'operation': 'list_all_domains',
            'time': operation_time,
            'success': True,
            'result_count': len(result)
        })
        
        return result
    
    def get_domain_count(self) -> int:
        """Retorna o número total de domínios no sistema."""
        return self.tree.size
    
    def get_tree_height(self) -> int:
        """Retorna a altura da árvore."""
        if hasattr(self.tree, 'get_height'):
            if self.tree_type == TreeType.AVL:
                return self.tree.get_height(self.tree.root)
            else:
                return self.tree.get_height()
        return 0
    
    def load_data_from_file(self, filename: str) -> int:
        """
        Carrega dados de um arquivo JSON gerado pelo DNSDataGenerator.
        
        Args:
            filename: Caminho para o arquivo JSON
            
        Returns:
            Número de domínios carregados
        """
        generator = DNSDataGenerator()
        domain_ip_pairs = generator.load_from_file(filename)
        
        loaded_count = 0
        for domain, ips in domain_ip_pairs:
            # Adicionar primeiro IP (cria o domínio)
            if ips and self.add_domain(domain, ips[0]):
                loaded_count += 1
                
                # Adicionar IPs adicionais
                for ip in ips[1:]:
                    self.add_ip_to_domain(domain, ip)
        
        return loaded_count
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas completas do sistema.
        
        Returns:
            Dicionário com estatísticas do sistema e da árvore
        """
        tree_stats = self.tree.get_stats()
        
        # Calcular estatísticas das operações
        operation_stats = self._calculate_operation_stats()
        
        return {
            'tree_type': self.tree_type.value,
            'total_operations': self.total_operations,
            'domain_count': self.get_domain_count(),
            'tree_height': self.get_tree_height(),
            'tree_stats': tree_stats,
            'operation_stats': operation_stats
        }
    
    def _calculate_operation_stats(self) -> Dict[str, Any]:
        """Calcula estatísticas das operações realizadas."""
        if not self.operation_history:
            return {}
        
        stats = {}
        operations = {}
        
        for op in self.operation_history:
            op_type = op['operation']
            if op_type not in operations:
                operations[op_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'success_count': 0,
                    'times': []
                }
            
            operations[op_type]['count'] += 1
            operations[op_type]['total_time'] += op['time']
            operations[op_type]['times'].append(op['time'])
            
            if op['success']:
                operations[op_type]['success_count'] += 1
        
        # Calcular médias e estatísticas
        for op_type, data in operations.items():
            stats[op_type] = {
                'count': data['count'],
                'total_time': data['total_time'],
                'avg_time': data['total_time'] / data['count'],
                'success_rate': data['success_count'] / data['count'],
                'min_time': min(data['times']),
                'max_time': max(data['times'])
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reseta todas as estatísticas do sistema."""
        self.tree.reset_stats()
        self.total_operations = 0
        self.operation_history = []
    
    def print_system_info(self) -> None:
        """Imprime informações do sistema."""
        stats = self.get_system_stats()
        
        print(f"=== Sistema DNS - {self.tree_type.value} ===")
        print(f"Domínios: {stats['domain_count']}")
        print(f"Altura da árvore: {stats['tree_height']}")
        print(f"Total de operações: {stats['total_operations']}")
        print()
        
        if stats['operation_stats']:
            print("Estatísticas de operações:")
            for op_type, op_stats in stats['operation_stats'].items():
                print(f"  {op_type}:")
                print(f"    Contagem: {op_stats['count']}")
                print(f"    Tempo médio: {op_stats['avg_time']:.6f}s")
                print(f"    Taxa de sucesso: {op_stats['success_rate']:.2%}")
                print()

def main():
    """Função principal para demonstração do sistema."""
    print("=== Demonstração do Sistema DNS ===\n")
    
    # Criar sistemas com ambas as árvores
    avl_system = DNSSystem(TreeType.AVL)
    rb_system = DNSSystem(TreeType.RED_BLACK)
    
    # Dados de teste
    test_domains = [
        ("google.com", "8.8.8.8"),
        ("facebook.com", "157.240.1.35"),
        ("github.com", "140.82.112.3"),
        ("stackoverflow.com", "151.101.1.69"),
        ("python.org", "151.101.32.223")
    ]
    
    print("Adicionando domínios de teste...")
    for domain, ip in test_domains:
        avl_system.add_domain(domain, ip)
        rb_system.add_domain(domain, ip)
    
    # Adicionar IPs adicionais
    avl_system.add_ip_to_domain("google.com", "8.8.4.4")
    rb_system.add_ip_to_domain("google.com", "8.8.4.4")
    
    # Testar buscas
    print("\nTestando buscas...")
    for domain, _ in test_domains:
        avl_result = avl_system.search_domain(domain)
        rb_result = rb_system.search_domain(domain)
        print(f"{domain}: AVL={avl_result}, RB={rb_result}")
    
    # Mostrar estatísticas
    print("\n" + "="*50)
    avl_system.print_system_info()
    print("="*50)
    rb_system.print_system_info()

if __name__ == "__main__":
    main()

