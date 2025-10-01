# Configuração da Pexels API

Este guia explica como obter e configurar sua chave de API da Pexels para a funcionalidade de visualização dinâmica de carros.

## Por que Pexels?

- 100% gratuito
- 20.000 requisições por mês no plano gratuito
- Mais de 100.000 fotos de carros de alta qualidade
- Sem necessidade de cartão de crédito
- API simples e bem documentada

## Passos para Configuração

### 1. Criar Conta na Pexels

1. Acesse: https://www.pexels.com/
2. Clique em "Sign Up" no canto superior direito
3. Crie sua conta (pode usar email ou conta Google/Facebook)

### 2. Obter Chave de API

1. Após fazer login, acesse: https://www.pexels.com/api/
2. Clique em "Get Started" ou "Your API Key"
3. Preencha o formulário com:
   - **Project Name**: "Predição de Preços de Veículos"
   - **Description**: "Aplicação educacional de Machine Learning para predição de preços de carros no mercado brasileiro"
   - **Project Type**: "Personal Project" ou "Educational"
4. Aceite os termos de uso
5. Clique em "Generate API Key"
6. Copie sua chave de API (formato: `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`)

### 3. Configurar no Projeto

1. Na raiz do projeto frontend, copie o arquivo `.env.example`:
   ```bash
   cd frontend
   cp .env.example .env
   ```

2. Abra o arquivo `.env` e adicione sua chave:
   ```
   VITE_API_URL=http://localhost:8000
   VITE_PEXELS_API_KEY=sua_chave_api_aqui
   ```

3. Salve o arquivo

### 4. Reiniciar o Servidor de Desenvolvimento

```bash
npm run dev
```

## Como Funciona

A aplicação usa a Pexels API para buscar imagens reais de carros baseadas na marca e modelo selecionados:

1. Usuário seleciona "Toyota" → Busca fotos de "Toyota"
2. Usuário seleciona "Corolla" → Busca fotos de "Toyota Corolla car"
3. Se encontrar imagens → Exibe foto real do carro
4. Se não encontrar → Exibe placeholder colorido com ícone

## Sistema de Fallback

A aplicação possui 3 níveis de fallback para garantir que sempre haja uma visualização:

```
1º Nível: Imagem real da Pexels API
   ↓ (se falhar)
2º Nível: Placeholder colorido dinâmico
   ↓ (se falhar)
3º Nível: Ícone SVG de carro
```

## Cache de Imagens

As imagens buscadas são armazenadas em cache por 10 minutos para:
- Reduzir requisições à API
- Melhorar performance
- Economizar cota mensal

## Limites da API Gratuita

- **50 requisições por hora** no modo demo
- **20.000 requisições por mês** após aprovação
- **Ilimitado** para projetos aprovados

Para projetos educacionais, geralmente 20.000 req/mês é mais que suficiente.

## Modo Demo (Sem API Key)

Se você não configurar a API key, a aplicação funcionará no modo demo:
- Sempre mostrará o placeholder colorido
- Não fará requisições à API
- Não haverá imagens reais de carros

## Troubleshooting

### Erro: "API key inválida"
- Verifique se copiou a chave corretamente
- Certifique-se de que não há espaços extras
- A chave deve ter exatamente 47 caracteres

### Imagens não carregam
- Verifique se o arquivo `.env` está na pasta `frontend/`
- Reinicie o servidor de desenvolvimento (`npm run dev`)
- Verifique o console do navegador para erros

### Erro: "Rate limit exceeded"
- Você excedeu o limite de 50 req/hora ou 20.000 req/mês
- Aguarde ou solicite aumento de cota no dashboard da Pexels

## Documentação Oficial

- Pexels API Docs: https://www.pexels.com/api/documentation/
- Rate Limits: https://www.pexels.com/api/documentation/#guidelines
- FAQ: https://help.pexels.com/

## Alternativas (Futuro)

Se precisar de mais imagens ou funcionalidades:
- **Unsplash API**: 50 req/hora gratuito
- **Pixabay API**: 100 req/minuto gratuito
- **FIPE API**: Dados de veículos brasileiros

## Suporte

Se tiver problemas:
1. Verifique os logs do console do navegador
2. Verifique os logs do terminal onde o frontend está rodando
3. Consulte a documentação oficial da Pexels
4. Abra uma issue no repositório do projeto
