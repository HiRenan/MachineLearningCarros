import React, { useState, useEffect } from 'react';
import { getMarcas, getModelos, getCores, getCambios, getCombustiveis, predictPrice } from '../services/api';

const PredictionForm = ({ onPrediction, onLoading }) => {
  const [formData, setFormData] = useState({
    marca: '',
    modelo: '',
    ano: '',
    quilometragem: '',
    cor: '',
    cambio: '',
    combustivel: '',
    portas: '4'
  });

  const [options, setOptions] = useState({
    marcas: [],
    modelos: {},
    cores: [],
    cambios: [],
    combustiveis: []
  });

  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const [marcas, modelos, cores, cambios, combustiveis] = await Promise.all([
          getMarcas(),
          getModelos(),
          getCores(),
          getCambios(),
          getCombustiveis()
        ]);

        setOptions({
          marcas,
          modelos,
          cores,
          cambios,
          combustiveis
        });
      } catch (error) {
        console.error('Erro ao carregar opções:', error);
      }
    };

    fetchOptions();
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value,
      ...(name === 'marca' && { modelo: '' })
    }));

    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }));
    }
  };

  const validateForm = () => {
    const newErrors = {};

    if (!formData.marca) newErrors.marca = 'Selecione uma marca';
    if (!formData.modelo) newErrors.modelo = 'Selecione um modelo';
    if (!formData.ano || formData.ano < 2000 || formData.ano > 2024) {
      newErrors.ano = 'Ano deve estar entre 2000 e 2024';
    }
    if (!formData.quilometragem || formData.quilometragem < 0) {
      newErrors.quilometragem = 'Quilometragem deve ser maior ou igual a zero';
    }
    if (!formData.cor) newErrors.cor = 'Selecione uma cor';
    if (!formData.cambio) newErrors.cambio = 'Selecione o tipo de câmbio';
    if (!formData.combustivel) newErrors.combustivel = 'Selecione o tipo de combustível';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setLoading(true);
    onLoading(true);

    try {
      const data = {
        marca: formData.marca,
        modelo: formData.modelo,
        ano: parseInt(formData.ano),
        quilometragem: parseFloat(formData.quilometragem),
        cor: formData.cor,
        cambio: formData.cambio,
        combustivel: formData.combustivel,
        portas: parseInt(formData.portas)
      };

      const prediction = await predictPrice(data);
      onPrediction(prediction);
    } catch (error) {
      console.error('Erro ao realizar predição:', error);
      alert('Erro ao realizar predição. Por favor, tente novamente.');
    } finally {
      setLoading(false);
      onLoading(false);
    }
  };

  const modelosDisponiveis = formData.marca ? options.modelos[formData.marca] || [] : [];

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <h2>Dados do Veículo</h2>

      <div className="form-grid">
        <div className="form-group">
          <label htmlFor="marca">Marca</label>
          <select
            id="marca"
            name="marca"
            value={formData.marca}
            onChange={handleChange}
            className={errors.marca ? 'error' : ''}
          >
            <option value="">Selecione a marca</option>
            {options.marcas.map(marca => (
              <option key={marca} value={marca}>{marca}</option>
            ))}
          </select>
          {errors.marca && <span className="error-message">{errors.marca}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="modelo">Modelo</label>
          <select
            id="modelo"
            name="modelo"
            value={formData.modelo}
            onChange={handleChange}
            disabled={!formData.marca}
            className={errors.modelo ? 'error' : ''}
          >
            <option value="">Selecione o modelo</option>
            {modelosDisponiveis.map(modelo => (
              <option key={modelo} value={modelo}>{modelo}</option>
            ))}
          </select>
          {errors.modelo && <span className="error-message">{errors.modelo}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="ano">Ano</label>
          <input
            type="number"
            id="ano"
            name="ano"
            value={formData.ano}
            onChange={handleChange}
            placeholder="Ex: 2020"
            min="2000"
            max="2024"
            className={errors.ano ? 'error' : ''}
          />
          {errors.ano && <span className="error-message">{errors.ano}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="quilometragem">Quilometragem (km)</label>
          <input
            type="number"
            id="quilometragem"
            name="quilometragem"
            value={formData.quilometragem}
            onChange={handleChange}
            placeholder="Ex: 45000"
            min="0"
            className={errors.quilometragem ? 'error' : ''}
          />
          {errors.quilometragem && <span className="error-message">{errors.quilometragem}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="cor">Cor</label>
          <select
            id="cor"
            name="cor"
            value={formData.cor}
            onChange={handleChange}
            className={errors.cor ? 'error' : ''}
          >
            <option value="">Selecione a cor</option>
            {options.cores.map(cor => (
              <option key={cor} value={cor}>{cor}</option>
            ))}
          </select>
          {errors.cor && <span className="error-message">{errors.cor}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="cambio">Câmbio</label>
          <select
            id="cambio"
            name="cambio"
            value={formData.cambio}
            onChange={handleChange}
            className={errors.cambio ? 'error' : ''}
          >
            <option value="">Selecione o câmbio</option>
            {options.cambios.map(cambio => (
              <option key={cambio} value={cambio}>{cambio}</option>
            ))}
          </select>
          {errors.cambio && <span className="error-message">{errors.cambio}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="combustivel">Combustível</label>
          <select
            id="combustivel"
            name="combustivel"
            value={formData.combustivel}
            onChange={handleChange}
            className={errors.combustivel ? 'error' : ''}
          >
            <option value="">Selecione o combustível</option>
            {options.combustiveis.map(combustivel => (
              <option key={combustivel} value={combustivel}>{combustivel}</option>
            ))}
          </select>
          {errors.combustivel && <span className="error-message">{errors.combustivel}</span>}
        </div>

        <div className="form-group">
          <label htmlFor="portas">Portas</label>
          <select
            id="portas"
            name="portas"
            value={formData.portas}
            onChange={handleChange}
          >
            <option value="2">2 portas</option>
            <option value="4">4 portas</option>
          </select>
        </div>
      </div>

      <button type="submit" className="submit-button" disabled={loading}>
        {loading ? (
          <>
            <span className="spinner"></span>
            Calculando...
          </>
        ) : (
          'Calcular Preço'
        )}
      </button>
    </form>
  );
};

export default PredictionForm;
