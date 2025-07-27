"""
Implementação de Árvore AVL
===========================

Esta implementação fornece uma árvore AVL balanceada automaticamente
para armazenar pares domínio-IP com operações de inserção,
busca e remoção.
"""

import time
from typing import List, Optional, Tuple, Any


class AVLNode:
    """Nó da árvore AVL contendo domínio e lista de IPs associados."""
    
    def __init__(self, domain: str, ip: str):
        self.domain = domain
        self.ips = [ip]  # Lista de IPs para o domínio
        self.height = 1
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
    
    def add_ip(self, ip: str) -> bool:
        """Adiciona um IP ao domínio se não existir."""
        if ip not in self.ips:
            self.ips.append(ip)
            return True
        return False
    
    def remove_ip(self, ip: str) -> bool:
        """Remove um IP do domínio."""
        if ip in self.ips:
            self.ips.remove(ip)
            return True
        return False


class AVLTree:   
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self.size = 0
        
        # Estatísticas de operações
        self.insert_time = 0.0
        self.search_time = 0.0
        self.delete_time = 0.0
        self.operation_count = {'insert': 0, 'search': 0, 'delete': 0}
        
        # Contadores de rotações
        self.left_rotations = 0
        self.right_rotations = 0
    
    def get_height(self, node: Optional[AVLNode]) -> int:
        """Retorna a altura do nó."""
        if not node:
            return 0
        return node.height
    
    def get_balance(self, node: Optional[AVLNode]) -> int:
        """Calcula o fator de balanceamento do nó."""
        if not node:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)
    
    def update_height(self, node: AVLNode) -> None:
        """Atualiza a altura do nó baseada nas alturas dos filhos."""
        node.height = 1 + max(self.get_height(node.left), 
                             self.get_height(node.right))
    
    def rotate_right(self, y: AVLNode) -> AVLNode:
        """Rotação à direita para balanceamento."""
        self.right_rotations += 1
        x = y.left
        T2 = x.right
        
        # Realizar rotação
        x.right = y
        y.left = T2
        
        # Atualizar alturas
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    def rotate_left(self, x: AVLNode) -> AVLNode:
        """Rotação à esquerda para balanceamento."""
        self.left_rotations += 1
        y = x.right
        T2 = y.left
        
        # Realizar rotação
        y.left = x
        x.right = T2
        
        # Atualizar alturas
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, domain: str, ip: str) -> bool:
        """
        Insere um domínio com IP na árvore.
        
        Args:
            domain: Nome do domínio
            ip: Endereço IP associado
            
        Returns:
            True se inserção foi bem-sucedida, False se domínio já existe
        """
        start_time = time.perf_counter()
        
        result, self.root = self._insert_recursive(self.root, domain, ip)
        
        end_time = time.perf_counter()
        self.insert_time += (end_time - start_time)
        self.operation_count['insert'] += 1
        
        if result:
            self.size += 1
        
        return result
    
    def _insert_recursive(self, node: Optional[AVLNode], domain: str, ip: str) -> Tuple[bool, AVLNode]:
        """Inserção recursiva com balanceamento."""
        # Caso base: inserir novo nó
        if not node:
            return True, AVLNode(domain, ip)
        
        # Inserção normal de BST
        if domain < node.domain:
            result, node.left = self._insert_recursive(node.left, domain, ip)
        elif domain > node.domain:
            result, node.right = self._insert_recursive(node.right, domain, ip)
        else:
            # Domínio já existe, adicionar IP
            result = node.add_ip(ip)
        
        if not result:
            return False, node
        
        # Atualizar altura
        self.update_height(node)
        
        # Obter fator de balanceamento
        balance = self.get_balance(node)
        
        # Casos de rotação
        # Caso Left Left
        if balance > 1 and domain < node.left.domain:
            return True, self.rotate_right(node)
        
        # Caso Right Right
        if balance < -1 and domain > node.right.domain:
            return True, self.rotate_left(node)
        
        # Caso Left Right
        if balance > 1 and domain > node.left.domain:
            node.left = self.rotate_left(node.left)
            return True, self.rotate_right(node)
        
        # Caso Right Left
        if balance < -1 and domain < node.right.domain:
            node.right = self.rotate_right(node.right)
            return True, self.rotate_left(node)
        
        return True, node
    
    def search(self, domain: str) -> Optional[List[str]]:
        """
        Busca um domínio na árvore.
        
        Args:
            domain: Nome do domínio a buscar
            
        Returns:
            Lista de IPs associados ao domínio ou None se não encontrado
        """
        start_time = time.perf_counter()
        
        result = self._search_recursive(self.root, domain)
        
        end_time = time.perf_counter()
        self.search_time += (end_time - start_time)
        self.operation_count['search'] += 1
        
        return result
    
    def _search_recursive(self, node: Optional[AVLNode], domain: str) -> Optional[List[str]]:
        """Busca recursiva na árvore."""
        if not node:
            return None
        
        if domain == node.domain:
            return node.ips.copy()
        elif domain < node.domain:
            return self._search_recursive(node.left, domain)
        else:
            return self._search_recursive(node.right, domain)
    
    def delete(self, domain: str) -> bool:
        """
        Remove um domínio da árvore.
        
        Args:
            domain: Nome do domínio a remover
            
        Returns:
            True se remoção foi bem-sucedida, False se domínio não existe
        """
        start_time = time.perf_counter()
        
        result, self.root = self._delete_recursive(self.root, domain)
        
        end_time = time.perf_counter()
        self.delete_time += (end_time - start_time)
        self.operation_count['delete'] += 1
        
        if result:
            self.size -= 1
        
        return result
    
    def _delete_recursive(self, node: Optional[AVLNode], domain: str) -> Tuple[bool, Optional[AVLNode]]:
        """Remoção recursiva com balanceamento."""
        if not node:
            return False, None
        
        # Remoção normal de BST
        if domain < node.domain:
            result, node.left = self._delete_recursive(node.left, domain)
        elif domain > node.domain:
            result, node.right = self._delete_recursive(node.right, domain)
        else:
            # Nó encontrado
            result = True
            
            # Caso 1: Nó folha ou com um filho
            if not node.left:
                return True, node.right
            elif not node.right:
                return True, node.left
            
            # Caso 2: Nó com dois filhos
            # Encontrar sucessor inorder (menor valor na subárvore direita)
            successor = self._find_min(node.right)
            
            # Copiar dados do sucessor
            node.domain = successor.domain
            node.ips = successor.ips.copy()
            
            # Remover sucessor
            _, node.right = self._delete_recursive(node.right, successor.domain)
        
        if not result:
            return False, node
        
        # Atualizar altura
        self.update_height(node)
        
        # Obter fator de balanceamento
        balance = self.get_balance(node)
        
        # Casos de rotação
        # Caso Left Left
        if balance > 1 and self.get_balance(node.left) >= 0:
            return True, self.rotate_right(node)
        
        # Caso Left Right
        if balance > 1 and self.get_balance(node.left) < 0:
            node.left = self.rotate_left(node.left)
            return True, self.rotate_right(node)
        
        # Caso Right Right
        if balance < -1 and self.get_balance(node.right) <= 0:
            return True, self.rotate_left(node)
        
        # Caso Right Left
        if balance < -1 and self.get_balance(node.right) > 0:
            node.right = self.rotate_right(node.right)
            return True, self.rotate_left(node)
        
        return True, node
    
    def _find_min(self, node: AVLNode) -> AVLNode:
        """Encontra o nó com valor mínimo na subárvore."""
        while node.left:
            node = node.left
        return node
    
    def add_ip_to_domain(self, domain: str, ip: str) -> bool:
        """Adiciona um IP a um domínio existente."""
        node = self._find_node(self.root, domain)
        if node:
            return node.add_ip(ip)
        return False
    
    def remove_ip_from_domain(self, domain: str, ip: str) -> bool:
        """Remove um IP de um domínio."""
        node = self._find_node(self.root, domain)
        if node:
            result = node.remove_ip(ip)
            # Se não há mais IPs, remover o domínio
            if not node.ips:
                self.delete(domain)
            return result
        return False
    
    def _find_node(self, node: Optional[AVLNode], domain: str) -> Optional[AVLNode]:
        """Encontra um nó específico na árvore."""
        if not node:
            return None
        
        if domain == node.domain:
            return node
        elif domain < node.domain:
            return self._find_node(node.left, domain)
        else:
            return self._find_node(node.right, domain)
    
    def get_all_domains(self) -> List[Tuple[str, List[str]]]:
        """Retorna todos os domínios e seus IPs."""
        domains = []
        self._inorder_traversal(self.root, domains)
        return domains
    
    def _inorder_traversal(self, node: Optional[AVLNode], domains: List[Tuple[str, List[str]]]) -> None:
        """Percurso inorder para coletar todos os domínios."""
        if node:
            self._inorder_traversal(node.left, domains)
            domains.append((node.domain, node.ips.copy()))
            self._inorder_traversal(node.right, domains)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas de desempenho da árvore."""
        stats = {
            'size': self.size,
            'height': self.get_height(self.root),
            'total_insert_time': self.insert_time,
            'total_search_time': self.search_time,
            'total_delete_time': self.delete_time,
            'avg_insert_time': self.insert_time / max(1, self.operation_count['insert']),
            'avg_search_time': self.search_time / max(1, self.operation_count['search']),
            'avg_delete_time': self.delete_time / max(1, self.operation_count['delete']),
            'operation_count': self.operation_count.copy()
        }
        stats.update(self.get_rotation_stats())
        return stats
    
    def get_rotation_stats(self) -> dict:
        """Retorna o número de rotações realizadas."""
        return {
            'left_rotations': self.left_rotations,
            'right_rotations': self.right_rotations,
            'total_rotations': self.left_rotations + self.right_rotations
        }
    
    def reset_stats(self) -> None:
        """Reseta as estatísticas de tempo."""
        self.insert_time = 0.0
        self.search_time = 0.0
        self.delete_time = 0.0
        self.operation_count = {'insert': 0, 'search': 0, 'delete': 0}
        self.left_rotations = 0
        self.right_rotations = 0

