import axios from 'axios';

const api = axios.create({
  baseURL: '/api'
});

export const listPapers = async () => {
  const { data } = await api.get('/papers');
  return data;
};

export const askQuestion = async (question, indexName, history = []) => {
  const { data } = await api.post('/ask', { question, index_name: indexName, history });
  return data;
};

export const ingestPaper = async (file, indexName) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('index_name', indexName);
  
  const { data } = await api.post('/papers/ingest', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
  return data;
};

export const deletePaper = async (name) => {
  const { data } = await api.delete(`/papers/${name}`);
  return data;
};

export const fetchArxiv = async (arxivId) => {
  const { data } = await api.post('/arxiv/fetch', { arxiv_id: arxivId });
  return data;
};

export const getPdfUrl = (name) => {
  return `/api/papers/${name}/pdf`;
};
