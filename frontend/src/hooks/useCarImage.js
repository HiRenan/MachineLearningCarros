import { useState, useEffect } from 'react';
import axios from 'axios';

const PEXELS_API_KEY = import.meta.env.VITE_PEXELS_API_KEY || 'demo';
const CACHE_DURATION = 1000 * 60 * 10; // 10 minutos

const imageCache = new Map();

export const useCarImage = (marca, modelo, cor) => {
  const [state, setState] = useState({
    type: 'loading',
    data: null,
    error: null
  });

  useEffect(() => {
    if (!marca || !modelo) {
      setState({ type: 'placeholder', data: cor, error: null });
      return;
    }

    const cacheKey = `${marca}-${modelo}`;
    const cachedData = imageCache.get(cacheKey);

    if (cachedData && Date.now() - cachedData.timestamp < CACHE_DURATION) {
      setState({ type: 'image', data: cachedData.url, error: null });
      return;
    }

    const fetchImage = async () => {
      setState({ type: 'loading', data: null, error: null });

      try {
        const searchQuery = `${marca} ${modelo} car`;
        const response = await axios.get('https://api.pexels.com/v1/search', {
          params: {
            query: searchQuery,
            per_page: 1,
            orientation: 'landscape'
          },
          headers: {
            Authorization: PEXELS_API_KEY
          },
          timeout: 5000
        });

        if (response.data.photos && response.data.photos.length > 0) {
          const imageUrl = response.data.photos[0].src.large;

          imageCache.set(cacheKey, {
            url: imageUrl,
            timestamp: Date.now()
          });

          setState({ type: 'image', data: imageUrl, error: null });
        } else {
          setState({ type: 'placeholder', data: cor, error: null });
        }
      } catch (error) {
        console.error('Erro ao buscar imagem do carro:', error);
        setState({ type: 'placeholder', data: cor, error: error.message });
      }
    };

    fetchImage();
  }, [marca, modelo, cor]);

  return state;
};
