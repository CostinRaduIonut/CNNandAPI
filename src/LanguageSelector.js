// import React, { useContext } from 'react';
// import { LanguageContext, availableLanguages } from './LanguageContext';
// const LanguageSelector = () => {
//   const { language, setLanguage } = useContext(LanguageContext);

//   const handleLanguageChange = (event) => {
//     const selectedLanguage = event.target.value;
//     setLanguage(selectedLanguage);
//   };

//   return (
//     <div>
//       <label htmlFor="language-select">Selectează limba:</label>
//       <select id="language-select" value={language} onChange={handleLanguageChange}>
//         {availableLanguages.map((lang) => (
//           <option key={lang} value={lang}>
//             {lang}
//           </option>
//         ))}
//       </select>
//     </div>
//   );
// };

// export default LanguageSelector;
import React, { useContext } from 'react';
import { LanguageContext } from './LanguageContext';

const LanguageSelector = () => {
  const { selectedLanguage, changeLanguage } = useContext(LanguageContext);

  const handleChangeLanguage = (event) => {
    const selectedLang = event.target.value;
    changeLanguage(selectedLang);
  };

  return (
    <div>
      <label htmlFor="language-select">Select Language:</label>
      <select id="language-select" value={selectedLanguage} onChange={handleChangeLanguage}>
        <option value="română">Română</option>
        <option value="english">English</option>
      </select>
    </div>
  );
};

export default LanguageSelector;
