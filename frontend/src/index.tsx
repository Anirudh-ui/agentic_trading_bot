// frontend/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // This imports your App.tsx component, as it's in the same folder
import './styles/globals.css'; // Adjust the path if your main CSS file is named differently

// Find the root element from public/index.html and render the App
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);