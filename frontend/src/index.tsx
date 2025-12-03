// frontend/index.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App'; // This imports your App.tsx component, as it's in the same folder
 // Adjust the path if your main CSS file is named differently
import './index.css'; // Import global styles
// Find the root element from public/index.html and render the App
ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
console.log(document.getElementById('root'))