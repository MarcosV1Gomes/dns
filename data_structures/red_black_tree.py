"""
Implementação de Árvore Rubro-Negra
===================================

Esta implementação fornece uma árvore rubro-negra balanceada automaticamente
para armazenar pares domínio-IP com operações de inserção,
busca e remoção.
"""

import time
from typing import List, Optional, Tuple
from enum import Enum


class Color(Enum):
    """Cores dos nós da árvore rubro-negra."""
    RED = "RED"
    BLACK = "BLACK"


class RBNode:
    """Nó da árvore rubro-negra contendo domínio e lista de IPs associados."""
    
    def __init__(self, domain: str, ip: str, color: Color = Color.RED):
        self.domain = domain
        self.ips = [ip]  # Lista de IPs para o domínio
        self.color = color
        self.left: Optional['RBNode'] = None
        self.right: Optional['RBNode'] = None
        self.parent: Optional['RBNode'] = None
    
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


class RedBlackTree:  
    def __init__(self):
        # Nó sentinela NIL (sempre preto)
        self.NIL = RBNode("", "", Color.BLACK)
        self.root = self.NIL
        self.size = 0
        
        # Estatísticas de operações
        self.insert_time = 0.0
        self.search_time = 0.0
        self.delete_time = 0.0
        self.operation_count = {'insert': 0, 'search': 0, 'delete': 0}
        # Contadores de rotações
        self.left_rotations = 0
        self.right_rotations = 0
    
    def rotate_left(self, x: RBNode) -> None:
        """Rotação à esquerda."""
        self.left_rotations += 1
        y = x.right
        x.right = y.left
        
        if y.left != self.NIL:
            y.left.parent = x
        
        y.parent = x.parent
        
        if x.parent == self.NIL:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        
        y.left = x
        x.parent = y
    
    def rotate_right(self, y: RBNode) -> None:
        """Rotação à direita."""
        self.right_rotations += 1
        x = y.left
        y.left = x.right
        
        if x.right != self.NIL:
            x.right.parent = y
        
        x.parent = y.parent
        
        if y.parent == self.NIL:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        
        x.right = y
        y.parent = x
    
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
        
        # Verificar se domínio já existe
        existing_node = self._search_node(domain)
        if existing_node != self.NIL:
            result = existing_node.add_ip(ip)
            end_time = time.perf_counter()
            self.insert_time += (end_time - start_time)
            self.operation_count['insert'] += 1
            return result
        
        # Criar novo nó
        new_node = RBNode(domain, ip, Color.RED)
        new_node.left = self.NIL
        new_node.right = self.NIL
        new_node.parent = self.NIL
        
        # Inserção normal de BST
        y = self.NIL
        x = self.root
        
        while x != self.NIL:
            y = x
            if new_node.domain < x.domain:
                x = x.left
            else:
                x = x.right
        
        new_node.parent = y
        
        if y == self.NIL:
            self.root = new_node
        elif new_node.domain < y.domain:
            y.left = new_node
        else:
            y.right = new_node
        
        # Corrigir propriedades da árvore rubro-negra
        self._insert_fixup(new_node)
        
        self.size += 1
        
        end_time = time.perf_counter()
        self.insert_time += (end_time - start_time)
        self.operation_count['insert'] += 1
        
        return True
    
    def _insert_fixup(self, z: RBNode) -> None:
        """Corrige as propriedades da árvore após inserção."""
        while z.parent.color == Color.RED:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                
                if y.color == Color.RED:
                    # Caso 1: Tio é vermelho
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        # Caso 2: z é filho direito
                        z = z.parent
                        self.rotate_left(z)
                    
                    # Caso 3: z é filho esquerdo
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self.rotate_right(z.parent.parent)
            else:
                y = z.parent.parent.left
                
                if y.color == Color.RED:
                    # Caso 1: Tio é vermelho
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        # Caso 2: z é filho esquerdo
                        z = z.parent
                        self.rotate_right(z)
                    
                    # Caso 3: z é filho direito
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self.rotate_left(z.parent.parent)
        
        self.root.color = Color.BLACK
    
    def search(self, domain: str) -> Optional[List[str]]:
        """
        Busca um domínio na árvore.
        
        Args:
            domain: Nome do domínio a buscar
            
        Returns:
            Lista de IPs associados ao domínio ou None se não encontrado
        """
        start_time = time.perf_counter()
        
        node = self._search_node(domain)
        result = node.ips.copy() if node != self.NIL else None
        
        end_time = time.perf_counter()
        self.search_time += (end_time - start_time)
        self.operation_count['search'] += 1
        
        return result
    
    def _search_node(self, domain: str) -> RBNode:
        """Busca um nó na árvore."""
        x = self.root
        
        while x != self.NIL and domain != x.domain:
            if domain < x.domain:
                x = x.left
            else:
                x = x.right
        
        return x
    
    def delete(self, domain: str) -> bool:
        """
        Remove um domínio da árvore.
        
        Args:
            domain: Nome do domínio a remover
            
        Returns:
            True se remoção foi bem-sucedida, False se domínio não existe
        """
        start_time = time.perf_counter()
        
        z = self._search_node(domain)
        if z == self.NIL:
            end_time = time.perf_counter()
            self.delete_time += (end_time - start_time)
            self.operation_count['delete'] += 1
            return False
        
        self._delete_node(z)
        self.size -= 1
        
        end_time = time.perf_counter()
        self.delete_time += (end_time - start_time)
        self.operation_count['delete'] += 1
        
        return True
    
    def _delete_node(self, z: RBNode) -> None:
        """Remove um nó da árvore."""
        y = z
        y_original_color = y.color
        
        if z.left == self.NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right == self.NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            
            if y.parent == z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
        
        if y_original_color == Color.BLACK:
            self._delete_fixup(x)
    
    def _transplant(self, u: RBNode, v: RBNode) -> None:
        """Substitui a subárvore u pela subárvore v."""
        if u.parent == self.NIL:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        
        v.parent = u.parent
    
    def _minimum(self, x: RBNode) -> RBNode:
        """Encontra o nó mínimo na subárvore."""
        while x.left != self.NIL:
            x = x.left
        return x
    
    def _delete_fixup(self, x: RBNode) -> None:
        """Corrige as propriedades da árvore após remoção."""
        while x != self.root and x.color == Color.BLACK:
            if x == x.parent.left:
                w = x.parent.right
                
                if w.color == Color.RED:
                    # Caso 1: Irmão é vermelho
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self.rotate_left(x.parent)
                    w = x.parent.right
                
                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    # Caso 2: Ambos filhos do irmão são pretos
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right.color == Color.BLACK:
                        # Caso 3: Filho esquerdo do irmão é vermelho, direito é preto
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self.rotate_right(w)
                        w = x.parent.right
                    
                    # Caso 4: Filho direito do irmão é vermelho
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self.rotate_left(x.parent)
                    x = self.root
            else:
                w = x.parent.left
                
                if w.color == Color.RED:
                    # Caso 1: Irmão é vermelho
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self.rotate_right(x.parent)
                    w = x.parent.left
                
                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    # Caso 2: Ambos filhos do irmão são pretos
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left.color == Color.BLACK:
                        # Caso 3: Filho direito do irmão é vermelho, esquerdo é preto
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self.rotate_left(w)
                        w = x.parent.left
                    
                    # Caso 4: Filho esquerdo do irmão é vermelho
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self.rotate_right(x.parent)
                    x = self.root
        
        x.color = Color.BLACK
    
    def add_ip_to_domain(self, domain: str, ip: str) -> bool:
        """Adiciona um IP a um domínio existente."""
        node = self._search_node(domain)
        if node != self.NIL:
            return node.add_ip(ip)
        return False
    
    def remove_ip_from_domain(self, domain: str, ip: str) -> bool:
        """Remove um IP de um domínio."""
        node = self._search_node(domain)
        if node != self.NIL:
            result = node.remove_ip(ip)
            # Se não há mais IPs, remover o domínio
            if not node.ips:
                self.delete(domain)
            return result
        return False
    
    def get_all_domains(self) -> List[Tuple[str, List[str]]]:
        """Retorna todos os domínios e seus IPs."""
        domains = []
        self._inorder_traversal(self.root, domains)
        return domains
    
    def _inorder_traversal(self, node: RBNode, domains: List[Tuple[str, List[str]]]) -> None:
        """Percurso inorder para coletar todos os domínios."""
        if node != self.NIL:
            self._inorder_traversal(node.left, domains)
            domains.append((node.domain, node.ips.copy()))
            self._inorder_traversal(node.right, domains)
    
    def get_height(self) -> int:
        """Calcula a altura da árvore."""
        return self._calculate_height(self.root)
    
    def _calculate_height(self, node: RBNode) -> int:
        """Calcula recursivamente a altura de um nó."""
        if node == self.NIL:
            return 0
        
        left_height = self._calculate_height(node.left)
        right_height = self._calculate_height(node.right)
        
        return 1 + max(left_height, right_height)
    
    def get_stats(self) -> dict:
        """Retorna estatísticas de desempenho da árvore."""
        stats = {
            'size': self.size,
            'height': self.get_height(),
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

