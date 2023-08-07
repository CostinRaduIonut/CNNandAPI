import React from 'react';
import { BrowserRouter as Router, Switch, Route,  BrowserRouter, Routes } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import Conversie from './Conversie';
import Teorie from './Teorie';
import Pastrate from './Pastrate';
import Home from './Home';
import Termeni from './Termeni';
import { LanguageProvider } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
function App() {


  return (
    <BrowserRouter>
      <LanguageProvider>
      <LanguageSelector />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Conversie" element={<Conversie />} />
          <Route path="/Teorie" element={<Teorie />} />
          <Route path="/Pastrate" element={<Pastrate />} />
          <Route path="/Termeni" element={<Termeni />} />
        </Routes>
      </LanguageProvider>
      <ToastContainer></ToastContainer>
    </BrowserRouter>
  );
}

export default App;
