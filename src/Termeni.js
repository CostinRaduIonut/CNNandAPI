import {  useNavigate } from 'react-router-dom';
import React, {  useContext } from 'react';
import { LanguageContext } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
import './Home.css';
import './Include.css'
const Termeni = () => {
    const { selectedLanguage } = useContext(LanguageContext);

    //navigare
    const navigate = useNavigate()
    const toAcasa = () => {
        navigate("/")
    }
    const toConversie = () => {
        navigate("/Conversie")
    }
    const toTeorie = () => {
        navigate("/Teorie")
    }
  
    const toTermeni = () => {
        navigate("/Termeni")
    }
    return (
        <>
            {selectedLanguage === 'română' ? (<>
                <div className="fixed-nav sticky-footer bg-dark" id="page-top">
                    <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top" id="mainNav">
                        <a className="navbar-brand" href="#">Convertor</a>
                        <LanguageSelector />
                        <button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">

                            <span className="navbar-toggler-icon"></span>
                        </button>
                        <div className="collapse navbar-collapse " id="navbarResponsive">
                            <ul className="navbar-nav navbar-sidenav " id="exampleAccordion">
                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Acasa">
                                    <a className="nav-link" onClick={toAcasa}>
                                        <i className="fa fa-home"></i>
                                        <span className="nav-link-text"> Acasă</span>
                                    </a>
                                </li>
                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Îngrijire">
                                    <a className="nav-link" onClick={toConversie}>
                                        <i className="fa fa-exchange"></i>
                                        <span className="nav-link-text"> Conversie</span>
                                    </a>
                                </li>

                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Legislație">
                                    <a className="nav-link nav-link-collapse collapsed" data-toggle="collapse" href="#collapseMulti" data-parent="#exampleAccordion" onClick={toTeorie}>
                                        <i className="fa fa-book"></i>
                                        <span className="nav-link-text"> Teorie</span>
                                    </a>
                                    <ul className="sidenav-second-level collapse" id="collapseMulti">
                                        <li>
                                            <a className="nav-link-collapse collapsed" data-toggle="collapse" href="#collapseMulti2">Reguli</a>
                                            <ul className="sidenav-third-level collapse" id="collapseMulti2">
                                                <li>
                                                    <a href="#unu"> Alfabet </a>
                                                </li>
                                                <li>
                                                    <a href="#doi"> Cifre | Litere</a>
                                                </li>
                                                <li>
                                                    <a href="#trei"> Punctuație</a>
                                                </li>
                                                <li>
                                                    <a href="#patru"> Istoric</a>
                                                </li>
                                            </ul>
                                        </li>
                                    </ul>
                                </li>

                            </ul>
                        </div>
                    </nav>
                    <div className="content-wrapper">
                        <div className="container-fluid">
                            {/* <!-- Breadcrumbs--> */}
                            {/* <ol className="breadcrumb">
                                <li className="breadcrumb-item">
                                    <a href="#">Termenii și condițiile</a>
                                </li>
                                <li className="breadcrumb-item active">Termenii și condițiile</li>
                            </ol> */}
                            <h1>Termenii și condițiile</h1>
                            <h2>Bun venit la Convertorul Braille și audio</h2>

                            Prelucrarea adresei IP<br></br>
                            1.1. În timp ce utilizați site-ul nostru, adresa IP a dispozitivului dumneavoastră va fi colectată în scopuri de securitate și pentru a îmbunătăți serviciile noastre. Prin utilizarea acestui site, sunteți de acord cu prelucrarea adresei IP în conformitate cu politica noastră de confidențialitate.<br></br>

                            1.2. Adresa IP poate fi utilizată pentru a personaliza și a vă oferi o experiență mai bună pe site-ul nostru. De asemenea, adresa IP poate fi utilizată în scopuri de diagnosticare a problemelor tehnice sau pentru a identifica abuzuri sau activități ilegale.<br></br>

                            1.3. Ne angajăm să protejăm confidențialitatea adresei IP și să o utilizăm numai în conformitate cu politica noastră de confidențialitate. Nu vom partaja adresa IP cu terțe părți în afara cazurilor în care acest lucru este necesar pentru furnizarea serviciilor noastre sau în cazurile prevăzute de lege.<br></br>

                            Prelucrarea numelui și prenumelui<br></br>
                            2.1. Pentru a vă oferi anumite servicii personalizate, precum crearea unui cont sau participarea la anumite activități, vă vom solicita să furnizați numele și prenumele dumneavoastră.<br></br>

                            2.2. Numele și prenumele dumneavoastră vor fi prelucrate în conformitate cu politica noastră de confidențialitate și vor fi utilizate exclusiv în scopurile specificate la momentul colectării.<br></br>

                            2.3. Ne angajăm să protejăm confidențialitatea numelui și prenumelui dumneavoastră și să le utilizăm numai în scopurile prevăzute în acești termeni și condiții și în conformitate cu politica noastră de confidențialitate.<br></br>

                            2.4. Nu vom partaja numele și prenumele dumneavoastră cu terțe părți în afara cazurilor în care acest lucru este necesar pentru furnizarea serviciilor noastre sau în cazurile prevăzute de lege.<br></br>

                            Securitatea datelor<br></br>
                            3.1. Ne angajăm să luăm măsuri tehnice și organizaționale adecvate pentru a proteja datele dumneavoastră, inclusiv adresa IP, numele și prenumele, împotriva accesului neautorizat, pierderii sau distrugerii accidentale.<br></br>

                            3.2. Cu toate acestea, vă reamintim că nicio metodă de transmitere sau stocare electronică nu este 100% sigură, iar securitatea absolută nu poate fi garantată. Utilizarea site-ului nostru se face pe propriul dumneavoastră risc.<br></br>

                            Utilizarea responsabilă<br></br>
                            4.1. Vă rugăm să utilizați site-ul nostru într-un mod responsabil și în conformitate cu termenii și condițiile noastre. Nu vă angajați în activități care să încalce legile în vigoare sau să afecteze negativ securitatea sau integritatea site-ului nostru.<br></br>

                            4.2. Rețineți că orice utilizare abuzivă sau ilegală poate duce la dezvăluirea adresei IP, numelor și prenumelor și la luarea măsurilor legale necesare.
                        </div>
                    </div>
                    <footer className="sticky-footer">
                        <div className="container">
                            <div className="text-center">
                                ©Costin Radu Ionuț <a href="https://www.facebook.com/ionut.radu.5036/" className="fa fa-facebook"></a><a href="https://www.instagram.com/" className="fa fa-instagram"></a>
                                <a onClick={toTermeni} style={{ color: "black", float: "right" }}>Termenii și condițiile</a>
                            </div>
                        </div>
                    </footer>
                </div>
            </>) : (<>
                <div className="fixed-nav sticky-footer bg-dark" id="page-top">
                    <nav className="navbar navbar-expand-lg navbar-dark bg-dark fixed-top" id="mainNav">
                        <a className="navbar-brand" href="#">Converter</a>
                        <LanguageSelector />
                        <button className="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">

                            <span className="navbar-toggler-icon"></span>
                        </button>
                        <div className="collapse navbar-collapse " id="navbarResponsive">
                            <ul className="navbar-nav navbar-sidenav " id="exampleAccordion">
                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Acasa">
                                    <a className="nav-link" onClick={toAcasa}>
                                        <i className="fa fa-home"></i>
                                        <span className="nav-link-text"> Home</span>
                                    </a>
                                </li>
                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Îngrijire">
                                    <a className="nav-link" onClick={toConversie}>
                                        <i className="fa fa-exchange"></i>
                                        <span className="nav-link-text"> Convert</span>
                                    </a>
                                </li>

                                <li className="nav-item" data-toggle="tooltip" data-placement="right" title="Legislație">
                                    <a className="nav-link nav-link-collapse collapsed" data-toggle="collapse" href="#collapseMulti" data-parent="#exampleAccordion" onClick={toTeorie}>
                                        <i className="fa fa-book"></i>
                                        <span className="nav-link-text"> Theory</span>
                                    </a>
                                    <ul className="sidenav-second-level collapse" id="collapseMulti">
                                        <li>
                                            <a className="nav-link-collapse collapsed" data-toggle="collapse" href="#collapseMulti2">Rules</a>
                                            <ul className="sidenav-third-level collapse" id="collapseMulti2">
                                                <li>
                                                    <a href="#unu"> Alphabet</a>
                                                </li>
                                                <li>
                                                    <a href="#doi"> Figures | Letters</a>
                                                </li>
                                                <li>
                                                    <a href="#trei"> Punctuation</a>
                                                </li>
                                                <li>
                                                    <a href="#patru"> History</a>
                                                </li>
                                            </ul>
                                        </li>
                                    </ul>
                                </li>

                            </ul>
                        </div>
                    </nav>
                    <div className="content-wrapper">
                        <div className="container-fluid">
                            {/* <!-- Breadcrumbs--> */}
                            {/* <ol className="breadcrumb">
                                <li className="breadcrumb-item">
                                    <a href="#">Terms and Conditions</a>
                                </li>
                                <li className="breadcrumb-item active">Terms and Conditions</li>
                            </ol> */}
                            <h1>Terms and Conditions</h1>
                            <h2>Welcome to the Braille and Audio Converter</h2>

                            Processing of IP Address<br></br>
                            1.1. While using our website, the IP address of your device will be collected for security purposes and to improve our services. By using this site, you agree to the processing of your IP address in accordance with our privacy policy.<br></br>

                            1.2. The IP address may be used to personalize and provide you with a better experience on our website. Additionally, the IP address may be used for diagnosing technical issues or identifying abuse or illegal activities.<br></br>

                            1.3. We are committed to protecting the confidentiality of the IP address and using it only in accordance with our privacy policy. We will not share the IP address with third parties unless necessary for providing our services or as required by law.<br></br>

                            Processing of Name and Surname<br></br>
                            2.1. To provide you with certain personalized services, such as creating an account or participating in specific activities, we may ask you to provide your name and surname.<br></br>

                            2.2. Your name and surname will be processed in accordance with our privacy policy and used solely for the purposes specified at the time of collection.<br></br>

                            2.3. We are committed to protecting the confidentiality of your name and surname and using them only for the purposes stated in these terms and conditions and in accordance with our privacy policy.<br></br>

                            2.4. We will not share your name and surname with third parties unless necessary for providing our services or as required by law.<br></br>

                            Data Security<br></br>
                            3.1. We are committed to implementing appropriate technical and organizational measures to protect your data, including IP address, name, and surname, against unauthorized access, accidental loss, or destruction.<br></br>

                            3.2. However, please be reminded that no method of electronic transmission or storage is 100% secure, and absolute security cannot be guaranteed. Your use of our website is at your own risk.<br></br>

                            Responsible Use<br></br>
                            4.1. Please use our website responsibly and in accordance with our terms and conditions. Do not engage in activities that violate applicable laws or negatively affect the security or integrity of our website.<br></br>

                            4.2. Please note that any abusive or illegal use may result in the disclosure of IP addresses, names, and surnames and the initiation of necessary legal actions.
                        </div>
                    </div>
                    <footer className="sticky-footer">
                        <div className="container">
                            <div className="text-center">
                                ©Costin Radu Ionuț <a href="https://www.facebook.com/ionut.radu.5036/" className="fa fa-facebook"></a><a href="https://www.instagram.com/" className="fa fa-instagram"></a>
                                <a onClick={toTermeni} style={{ color: "black", float: "right" }}>Terms and Conditions</a>
                            </div>
                        </div>
                    </footer>
                </div>

            </>)}
        </>
    )
}
export default Termeni