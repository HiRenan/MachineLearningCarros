# Frontend - Predição de Preços de Veículos

Interface web desenvolvida com React + Vite para predição de valores de venda de veículos.

## Tecnologias

- React 18
- Vite 5
- Axios (requisições HTTP)
- React Icons (ícones)
- Pexels API (imagens de carros)
- CSS3 (gradientes, grid, flexbox, animações)

## Instalação

```bash
cd frontend
npm install
```

## Configuração

Crie um arquivo `.env` baseado no `.env.example`:

```bash
cp .env.example .env
```

Edite o arquivo `.env` e configure:

```
VITE_API_URL=http://localhost:8000
VITE_PEXELS_API_KEY=your_pexels_api_key_here
```

**Para obter sua chave Pexels API:**
1. Acesse https://www.pexels.com/api/
2. Crie uma conta gratuita
3. Gere sua API key
4. Consulte [PEXELS_SETUP.md](PEXELS_SETUP.md) para instruções detalhadas

## Execução

### Modo Desenvolvimento

```bash
npm run dev
```

Acesse: http://localhost:5173

### Build para Produção

```bash
npm run build
```

Os arquivos otimizados serão gerados na pasta `dist/`.

### Preview da Build

```bash
npm run preview
```

## Estrutura

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Disclaimer.jsx       # Aviso educacional
│   │   ├── PredictionForm.jsx   # Formulário de entrada
│   │   ├── ResultCard.jsx       # Resultado da predição
│   │   └── ModelInfo.jsx        # Informações do modelo ML
│   ├── services/
│   │   └── api.js               # Configuração Axios
│   ├── App.jsx                  # Componente principal
│   ├── App.css                  # Estilos principais
│   ├── index.css                # Estilos globais
│   └── main.jsx                 # Entry point
├── .env.example
├── package.json
└── vite.config.js
```

## Funcionalidades

### Visualização Dinâmica de Carros
- Imagens reais de carros (Pexels API)
- Atualização em tempo real ao mudar marca/modelo/cor
- Sistema de fallback inteligente:
  1. Imagem real do carro
  2. Placeholder colorido dinâmico
  3. Ícone SVG de fallback
- Animações suaves de transição
- Cache de imagens (10 minutos)

### Formulário de Predição
- Seleção de marca com atualização dinâmica de modelos
- Validação completa de campos
- Feedback visual de erros
- Loading state durante predição

### Exibição de Resultados
- Valor estimado destacado
- Intervalo de confiança (mínimo/máximo)
- Badge de confiança da predição
- Informações sobre MAE do modelo

### Informações do Modelo
- Algoritmo utilizado (Lasso Regression)
- Métricas de performance (R², MAE, RMSE)
- Total de features
- Descrição do modelo

### Design Responsivo
- Layout adaptável para desktop, tablet e mobile
- Grid system com CSS Grid
- Mobile-first approach
- Transições e animações suaves

## Componentes

### Disclaimer
Exibe o aviso educacional obrigatório no topo da página.

### PredictionForm
Formulário completo para entrada de dados do veículo com:
- Dropdowns dinâmicos
- Validação em tempo real
- Loading state
- Tratamento de erros

### ResultCard
Apresenta o resultado da predição com:
- Valor principal em destaque
- Intervalo de confiança
- Badge de confiança
- Informações adicionais

### ModelInfo
Exibe métricas e informações do modelo ML treinado.

## API Integration

O frontend se comunica com a API FastAPI através do Axios:

```javascript
// Fazer predição
const prediction = await predictPrice({
  marca: "Toyota",
  modelo: "Corolla",
  ano: 2020,
  quilometragem: 45000,
  cor: "Prata",
  cambio: "Automático",
  combustivel: "Flex",
  portas: 4
});
```

## Estilos

O projeto utiliza CSS puro com:
- Variáveis CSS para cores
- Gradientes modernos
- Sistema de grid responsivo
- Animações CSS
- Estados de hover e focus
- Media queries para responsividade

### Paleta de Cores

- Primary: `#667eea` → `#764ba2`
- Success: `#48bb78`
- Warning: `#ed8936`
- Error: `#fc8181`
- Gray scale: `#1a202c` → `#f7fafc`

## Build e Deploy

### Build

```bash
npm run build
```

### Deploy em Vercel

```bash
npm install -g vercel
vercel
```

### Deploy em Netlify

```bash
npm install -g netlify-cli
netlify deploy --prod
```

## Variáveis de Ambiente

- `VITE_API_URL`: URL base da API (default: http://localhost:8000)

## Performance

- Code splitting automático (Vite)
- Lazy loading de componentes
- Otimização de assets
- Minificação de CSS/JS
- Tree shaking

## Browser Support

- Chrome (últimas 2 versões)
- Firefox (últimas 2 versões)
- Safari (últimas 2 versões)
- Edge (últimas 2 versões)

## Desenvolvido por

Renan & Nathan | 2025
