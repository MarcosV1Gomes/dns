"""
Gerador de Dados DNS Realistas
==============================

Este módulo utiliza a biblioteca Faker para gerar domínios e endereços IP
públicos realistas seguindo uma distribuição específica para simular um ambiente
DNS real.

Distribuição de IPs por domínio:
- 40% dos domínios têm 1 IP
- 35% dos domínios têm 2 IPs
- 25% dos domínios têm 3 IPs

Todos os IPs gerados são endereços públicos válidos.
"""

import json
import random
from typing import List, Dict, Tuple
from faker import Faker

class DNSDataGenerator:
    """
    Gerador de dados DNS realistas usando Faker.
    
    Gera domínios e endereços IP seguindo distribuições realistas
    para simular um ambiente DNS real.
    """
    
    def __init__(self, seed: int = 42):
        """
        Inicializa o gerador com uma semente para reprodutibilidade.
        
        Args:
            seed: Semente para geração determinística
        """
        self.faker = Faker()
        Faker.seed(seed)
        random.seed(seed)
        
        # Extensões de domínio mais comuns
        self.common_tlds = [
            '.com', '.org', '.net', '.edu', '.gov', '.info', '.biz'
        ]
        
        # Pesos para distribuição de TLDs (mais realista)
        self.tld_weights = [
            0.55,  # .com (55%)
            0.15,  # .net (15%)
            0.10,  # .org (10%)
            0.07,  # .edu (7%)
            0.03,  # .gov (3%)
            0.05,  # .info (5%)
            0.05,  # .biz (5%)
        ]
    
    def generate_realistic_domain(self) -> str:
        """
        Gera um domínio realista.
        
        Returns:
            String contendo um domínio válido
        """
        # Diferentes tipos de domínios
        domain_types = [
            self._generate_company_domain,
            self._generate_personal_domain,
            self._generate_organization_domain,
            self._generate_tech_domain,
            self._generate_geographic_domain
        ]
        
        # Escolher tipo de domínio aleatoriamente
        domain_generator = random.choice(domain_types)
        base_domain = domain_generator()
        
        # Escolher TLD baseado nos pesos
        tld = random.choices(self.common_tlds, weights=self.tld_weights)[0]
        
        return f"{base_domain}{tld}"
    
    def _generate_company_domain(self) -> str:
        """Gera domínio de empresa."""
        company_name = self.faker.company().lower()
        # Remover caracteres especiais e espaços
        company_name = ''.join(c for c in company_name if c.isalnum())
        return company_name[:20]  # Limitar tamanho
    
    def _generate_personal_domain(self) -> str:
        """Gera domínio pessoal."""
        first_name = self.faker.first_name().lower()
        last_name = self.faker.last_name().lower()
        
        patterns = [
            f"{first_name}{last_name}",
            f"{first_name}.{last_name}",
            f"{first_name}-{last_name}",
            f"{first_name}{random.randint(1, 999)}",
            f"{last_name}{random.randint(1, 999)}"
        ]
        
        return random.choice(patterns)
    
    def _generate_organization_domain(self) -> str:
        """Gera domínio de organização."""
        org_types = ['foundation', 'institute', 'center', 'society', 'association']
        word = self.faker.word().lower()
        org_type = random.choice(org_types)
        
        patterns = [
            f"{word}-{org_type}",
            f"{word}{org_type}",
            f"{org_type}-{word}"
        ]
        
        return random.choice(patterns)
    
    def _generate_tech_domain(self) -> str:
        """Gera domínio relacionado à tecnologia."""
        tech_words = [
            'tech', 'dev', 'code', 'app', 'web', 'digital', 'cyber',
            'cloud', 'data', 'ai', 'ml', 'api', 'soft', 'sys'
        ]
        
        word = self.faker.word().lower()
        tech_word = random.choice(tech_words)
        
        patterns = [
            f"{word}{tech_word}",
            f"{tech_word}{word}",
            f"{word}-{tech_word}",
            f"{tech_word}-{word}"
        ]
        
        return random.choice(patterns)
    
    def _generate_geographic_domain(self) -> str:
        """Gera domínio baseado em localização."""
        city = self.faker.city().lower()
        # Remover caracteres especiais
        city = ''.join(c for c in city if c.isalnum())
        
        suffixes = ['online', 'local', 'city', 'region', 'area']
        suffix = random.choice(suffixes)
        
        patterns = [
            f"{city}",
            f"{city}-{suffix}",
            f"{city}{suffix}"
        ]
        
        return random.choice(patterns)[:25]  # Limitar tamanho
    
    def generate_realistic_ip(self) -> str:
        """
        Gera um endereço IP público realista.
        
        Returns:
            String contendo um endereço IP público válido
        """
        return self._generate_public_ip()
    
    def _generate_public_ip(self) -> str:
        """Gera IP público."""
        # Evitar ranges reservados
        while True:
            ip = self.faker.ipv4()
            octets = ip.split('.')
            first_octet = int(octets[0])
            second_octet = int(octets[1])
            
            # Evitar ranges privados e reservados
            if (first_octet == 10 or 
                (first_octet == 172 and 16 <= second_octet <= 31) or
                (first_octet == 192 and second_octet == 168) or
                first_octet >= 224):
                continue
            
            return ip
    
    def generate_domain_ip_pairs(self, num_domains: int) -> List[Tuple[str, List[str]]]:
        """
        Gera pares domínio-IP seguindo a distribuição especificada.
        
        Args:
            num_domains: Número de domínios a gerar
            
        Returns:
            Lista de tuplas (domínio, lista_de_ips)
        """
        domain_ip_pairs = []
        domains_generated = set()  # Para evitar duplicatas
        
        for _ in range(num_domains):
            # Gerar domínio único
            domain = self.generate_realistic_domain()
            while domain in domains_generated:
                domain = self.generate_realistic_domain()
            domains_generated.add(domain)
            
            # Determinar número de IPs baseado na distribuição
            rand = random.random()
            if rand < 0.4:  # 40% - 1 IP
                num_ips = 1
            elif rand < 0.75:  # 35% - 2 IPs
                num_ips = 2
            else:  # 25% - 3 IPs
                num_ips = 3
            
            # Gerar IPs únicos para o domínio
            ips = []
            ips_generated = set()
            
            for _ in range(num_ips):
                ip = self.generate_realistic_ip()
                while ip in ips_generated:
                    ip = self.generate_realistic_ip()
                ips_generated.add(ip)
                ips.append(ip)
            
            domain_ip_pairs.append((domain, ips))
        
        return domain_ip_pairs
    
    def save_to_file(self, domain_ip_pairs: List[Tuple[str, List[str]]], 
                     filename: str, format_type: str = 'json') -> None:
        """
        Salva os dados gerados em arquivo.
        
        Args:
            domain_ip_pairs: Lista de pares domínio-IP
            filename: Nome do arquivo
            format_type: Formato do arquivo ('json' ou 'txt')
        """
        if format_type == 'json':
            data = {
                'domains': [
                    {'domain': domain, 'ips': ips} 
                    for domain, ips in domain_ip_pairs
                ],
                'metadata': {
                    'total_domains': len(domain_ip_pairs),
                    'distribution': self._calculate_distribution(domain_ip_pairs)
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'txt':
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("# Dados DNS Gerados\n")
                f.write(f"# Total de domínios: {len(domain_ip_pairs)}\n")
                f.write("# Formato: domínio -> ip1,ip2,ip3...\n\n")
                
                for domain, ips in domain_ip_pairs:
                    f.write(f"{domain} -> {','.join(ips)}\n")
    
    def _calculate_distribution(self, domain_ip_pairs: List[Tuple[str, List[str]]]) -> Dict[str, float]:
        """Calcula a distribuição real de IPs por domínio."""
        ip_counts = {}
        
        for _, ips in domain_ip_pairs:
            count = len(ips)
            ip_counts[count] = ip_counts.get(count, 0) + 1
        
        total = len(domain_ip_pairs)
        distribution = {}
        
        for count, freq in ip_counts.items():
            distribution[f"{count}_ips"] = (freq / total) * 100
        
        return distribution
    
    def load_from_file(self, filename: str) -> List[Tuple[str, List[str]]]:
        """
        Carrega dados de um arquivo JSON.
        
        Args:
            filename: Nome do arquivo JSON
            
        Returns:
            Lista de pares domínio-IP
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [(item['domain'], item['ips']) for item in data['domains']]

def main():
    """Função principal para demonstração."""
    generator = DNSDataGenerator()
    
    # Gerar diferentes tamanhos de dados para testes
    sizes = [100, 1000, 10000]
    
    for size in sizes:
        print(f"Gerando {size} domínios...")
        
        # Gerar dados
        domain_ip_pairs = generator.generate_domain_ip_pairs(size)
        
        # Salvar em arquivos
        json_filename = f"dns_data_{size}.json"
        txt_filename = f"dns_data_{size}.txt"
        
        generator.save_to_file(domain_ip_pairs, json_filename, 'json')
        generator.save_to_file(domain_ip_pairs, txt_filename, 'txt')
        
        print(f"Dados salvos em {json_filename} e {txt_filename}")
        
        # Mostrar estatísticas
        distribution = generator._calculate_distribution(domain_ip_pairs)
        print("Distribuição de IPs por domínio:")
        for key, value in distribution.items():
            print(f"  {key}: {value:.1f}%")
        print()

if __name__ == "__main__":
    main()

