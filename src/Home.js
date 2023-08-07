import React, { useContext } from 'react';
import { LanguageContext } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
import './Home.css';
import './Include.css'

import { useNavigate } from 'react-router-dom';

function Home() {
    const { selectedLanguage } = useContext(LanguageContext);


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

                    {/* <!-- Navigation--> */}
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
                            {/* <!-- ======= Header ======= --> */}
                            <header id="header" className="fixed-top ">
                                <div className="container-fluid">


                                </div>
                            </header>
                            {/* <!-- End Header --> */}

                            {/* <!-- ======= Hero Section ======= --> */}
                            <section id="hero" className="d-flex flex-column justify-content-center">
                                <div className="container">
                                    <div className="row justify-content-center">
                                        <div className="col-xl-8">
                                            <h1>Convertor text - Braille & audio</h1>
                                            <h2>Susținem educația în rândul nevăzătorilor</h2>
                                            <a href="https://www.youtube.com/watch?v=EOO_U7tNbLQ" className="glightbox play-btn mb-4"></a>
                                        </div>
                                    </div>
                                </div>
                            </section>
                            {/* <!-- End Hero --> */}

                            <main id="main">

                                {/* <!-- ======= About Us Section ======= --> */}
                                <section id="about" className="about">
                                    <div className="container">

                                        <div className="section-title">

                                            <h2 className="h">Despre noi</h2>
                                            <p>
                                                Aplicația dorește să introducă o metodă gratuită de a oferi o șansă în plus la educație persoanelor din
                                                comunitatea de nevăzători prin posibilitatea de a converti text provenit din diverse medii în braille pentru a fi salvat sub formă de
                                                imagine, ce poate fi imprimată direct pe hârtie specială pentru Braille, sau ascultat drept audio-book.
                                            </p>
                                        </div>

                                        <div className="row content">

                                            <p>
                                                Vă rugăm să contribuiți la îmbunătățirea aplicației prin completarea următorului formular:
                                            </p>

                                            <div className="col-lg-6 pt-4 pt-lg-0">
                                                <a href='https://forms.gle/1qAPRMAiHKyopQEB6' target="_blank" rel="noopener noreferrer" style={{ fontSize: "40px" }}>Completează formularul </a>
                                            </div>
                                            <div className="col-lg-6 pt-4 pt-lg-0">
                                                <p>
                                                    Convertorul se laudă cu un număr de <span style={{ color: "green", fontSize: "20px" }}>{localStorage.getItem("unique")}</span> utilizatori unici
                                                </p>
                                            </div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End About Us Section --> */}

                                {/* <!-- ======= Services Section ======= --> */}
                                <section id="services" className="services">
                                    <div className="container">

                                        <div className="section-title">
                                            <h2 className="h">Servicii</h2>
                                            <p>Aici se oferă diferite posibilități dintre care amintim:</p>
                                        </div>

                                        <div className="row">
                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch">
                                                <div className="icon-box">
                                                    <div className="icon"> <i className="fa fa-heart"></i>
                                                    </div>
                                                    <h4><a href="">Interfață prietenoasa</a></h4>
                                                    <p>
                                                        Interfața cu utilizatorul a fost creată pentru a fi ușor de folosit de orice văzător care dorește
                                                        să o folosească pentru a ajuta pe cei în nevoie.
                                                    </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4 mt-md-0">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-book"></i></div>
                                                    <h4><a href="">Teorie</a></h4>
                                                    <p>Albabetul, cifrele și regulile de scriere explicate, împreună un mic tutorial și istoric despre braille. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4 mt-lg-0">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-exchange"></i></div>
                                                    <h4><a href="">Text - Braille & Audio-book</a></h4>
                                                    <p>Aplicația traduce text din alfabet latin fără caractere speciale, în braille pentru a fi imprimat sub formă de carte Braille sau să fie ascultat. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-check"></i></div>
                                                    <h4><a href="">Multiple opțiuni</a></h4>
                                                    <p>Se oferă posibilitatea de a păstra textul introdus și de a-l șterge ulterior, iar Aplicația este disponibilă în limba preferată de user.</p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-sort-amount-asc"></i></div>
                                                    <h4><a href="">Numeroare inputuri</a></h4>
                                                    <p>Aplicația poate converti text provenit de la tastatură, dintr-o imagine sau din înregistrare audio. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"> <i className="fa fa-balance-scale"></i>
                                                    </div>
                                                    <h4><a href="">Autocorector</a></h4>
                                                    <p>În eventualitatea în care se dorește verificarea textului, este implementat un autocorector, dar el funcționează doar în limba engleză.</p>
                                                </div>
                                            </div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Services Section --> */}



                                {/* <!--</main> ======= Counts Section ======= --> */}
                                <section id="counts" className="counts">
                                    <div className="container">

                                        <div className="text-center title">
                                            <h3 className="h">Ce am reușit până acum</h3>
                                        </div>

                                        <div className="row counters position-relative">

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-duration="1" className="purecounter"  style={{color:"white"}}>text - braille</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="521" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>text - audio</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="1463" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>autocorector</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="15" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>Înregistrare vocală</span>
                                                <p></p>
                                            </div>

                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Counts Section --> */}



                                {/* <!-- ======= Faq Section ======= --> */}
                                <section id="faq" className="faq">
                                    <div className="container-fluid">

                                        <div className="row">

                                            <div className="col-lg-7 d-flex flex-column justify-content-center align-items-stretch  order-2 order-lg-1">

                                                <div className="content">
                                                    <h3><strong>Întrebări </strong> adresate frecvent</h3>
                                                    <p>
                                                        Această secțiune a fost creată cu scopul de a anticipa întrebările utilizatorilor.
                                                    </p>
                                                </div>

                                                <div className="accordion-list">
                                                    <ul>
                                                        <li>
                                                            <a data-bs-toggle="collapse" className="collapse" data-bs-target="#accordion-list-1">Aplicația funcționează pe toate tipurile de Braille? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-1" className="collapse show" data-bs-parent=".accordion-list">
                                                                <p>
                                                                    Nu, aplicația va funcționa doar pe Braille Grade 1 standard, fără caractere speciale precum diacritice</p>
                                                            </div>
                                                        </li>

                                                        <li>
                                                            <a data-bs-toggle="collapse" data-bs-target="#accordion-list-2" className="collapsed">Ce tip de inputuri sunt permise? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-2" className="collapse" data-bs-parent=".accordion-list">
                                                                <p >
                                                                    Textul convertit poate fi preluat din imagini, de la tastaura sau din format audio înregistrat de aplicație. Completarea unui câmp va duce la dezactivarea altor metode                                                        </p>
                                                            </div>
                                                        </li>

                                                        <li>
                                                            <a data-bs-toggle="collapse" data-bs-target="#accordion-list-3" className="collapsed">De ce uneori durează mai mult pentru a afișa rezultatul? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-3" className="collapse" data-bs-parent=".accordion-list">
                                                                <p >
                                                                    Uneori rezultatul final este afișat mai târziu din cauza faptului că imaginea conține mult zgomot, are numeroase caractere apropiate, conține un font greu de detectat sau pixeli prea mulți.                                                        </p>
                                                            </div>
                                                        </li>

                                                    </ul>
                                                </div>

                                            </div>

                                            <div className="col-lg-5 align-items-stretch order-1 order-lg-2 img" id="intrebare"></div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Faq Section --> */}
                                <section id="contact" className="contact">
                                    <div className="container">
                                        <div className="section-title">
                                            <h2 className="h"><i className="fa fa-map-marker" aria-hidden="true"></i> Locație</h2>

                                        </div>
                                    </div>

                                    <div>

                                        <div className="gmap_canvas"><iframe width="100%" height="350" id="gmap_canvas" src="https://maps.google.com/maps?q=garii%203a,%20Baia%20mare&t=&z=13&ie=UTF8&iwloc=&output=embed" frameBorder="0" scrolling="no" marginHeight="0" marginWidth="0"></iframe><a href="https://123movies-org.net"></a><br></br>
                                            <a href="https://www.embedgooglemap.net"></a>
                                        </div>
                                    </div>

                                    {/* <!-- End #main --> */}

                                </section>

                            </main>
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
                            {/* <!-- ======= Header ======= --> */}
                            <header id="header" className="fixed-top ">
                                <div className="container-fluid">


                                </div>
                            </header>
                            {/* <!-- End Header --> */}

                            {/* <!-- ======= Hero Section ======= --> */}
                            <section id="hero" className="d-flex flex-column justify-content-center">
                                <div className="container">
                                    <div className="row justify-content-center">
                                        <div className="col-xl-8">
                                            <h1>Text-to-Braille & Speech Converter</h1>
                                            <h2>We support education for the visually impaired.</h2>
                                            <a href="https://www.youtube.com/watch?v=EOO_U7tNbLQ" className="glightbox play-btn mb-4"></a>
                                        </div>
                                    </div>
                                </div>
                            </section>
                            {/* <!-- End Hero --> */}

                            <main id="main">

                                {/* <!-- ======= About Us Section ======= --> */}
                                <section id="about" className="about">
                                    <div className="container">

                                        <div className="section-title">

                                            <h2 className="h">About us</h2>
                                            <p>
                                                The application aims to introduce a free method of providing an additional opportunity in education for individuals in the visually impaired community by enabling the conversion of text into Braille for saving as a document, printing in bas-relief directly on paper, or listening as an audio book.                                            </p>
                                        </div>

                                        <div className="row content">

                                            <p>
                                                Please contribute to the improvement of the application by filling out the following form:                                            </p>

                                            <div className="col-lg-6 pt-4 pt-lg-0">
                                                <a href='https://forms.gle/1qAPRMAiHKyopQEB6' target="_blank" rel="noopener noreferrer" style={{ fontSize: "40px" }}>Complete the form </a>
                                            </div>
                                            <div className="col-lg-6 pt-4 pt-lg-0">
                                                <p>
                                                    The application boasts a number of <span style={{ color: "green", fontSize: "20px" }}>{localStorage.getItem("unique")}</span> unique users.
                                                </p>
                                            </div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End About Us Section --> */}

                                {/* <!-- ======= Services Section ======= --> */}
                                <section id="services" className="services">
                                    <div className="container">

                                        <div className="section-title">
                                            <h2 className="h">Services</h2>
                                            <p>Various options are provided, including::</p>
                                        </div>

                                        <div className="row">
                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch">
                                                <div className="icon-box">
                                                    <div className="icon"> <i className="fa fa-heart"></i>
                                                    </div>
                                                    <h4><a href="">User-friendly interface</a></h4>
                                                    <p>
                                                    The user interface has been designed to be easily usable by any sighted individual who wishes to use it to assist those in need.
                                                    </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4 mt-md-0">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-book"></i></div>
                                                    <h4><a href="">Theory</a></h4>
                                                    <p>The Alphabet, Numbers, and Rules of Writing Explained, Along with a Brief Tutorial and History of Braille. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4 mt-lg-0">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-exchange"></i></div>
                                                    <h4><a href="">Text - Braille & Audio-book</a></h4>
                                                    <p>The application translates text from the Latin alphabet, excluding special characters, into Braille for printing as a Braille book or for audio playback. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-check"></i></div>
                                                    <h4><a href="">Multiple options</a></h4>
                                                    <p>The option to save, view and delete entered text is provided, and the application is available in the user's preferred language.</p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"><i className="fa fa-sort-amount-asc"></i></div>
                                                    <h4><a href="">Multiple inputs</a></h4>
                                                    <p>The application can convert text from keyboard input, from an image, or from audio recordings. </p>
                                                </div>
                                            </div>

                                            <div className="col-lg-4 col-md-6 d-flex align-items-stretch mt-4">
                                                <div className="icon-box">
                                                    <div className="icon"> <i className="fa fa-balance-scale"></i>
                                                    </div>
                                                    <h4><a href="">Autocorrect</a></h4>
                                                    <p>In the event that text verification is desired, an autocorrect feature is implemented, but it only functions in English.</p>
                                                </div>
                                            </div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Services Section --> */}



                                {/* <!--</main> ======= Counts Section ======= --> */}
                                <section id="counts" className="counts">
                                    <div className="container">

                                        <div className="text-center title">
                                            <h3 className="h">What I have accomplished so far:</h3>
                                        </div>

                                        <div className="row counters position-relative">

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>text-to-Braille</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="521" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>text-to-speech</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="1463" data-purecounter-duration="1" className="purecounter"style={{color:"white"}}> autocorrect</span>
                                                <p></p>
                                            </div>

                                            <div className="col-lg-3 col-6 text-center">
                                                <span data-purecounter-start="0" data-purecounter-end="15" data-purecounter-duration="1" className="purecounter" style={{color:"white"}}>voice recording</span>
                                                <p></p>
                                            </div>

                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Counts Section --> */}



                                {/* <!-- ======= Faq Section ======= --> */}
                                <section id="faq" className="faq">
                                    <div className="container-fluid">

                                        <div className="row">

                                            <div className="col-lg-7 d-flex flex-column justify-content-center align-items-stretch  order-2 order-lg-1">

                                                <div className="content">
                                                    <h3>Frequently Asked<strong> Questions</strong> (FAQ)</h3>
                                                    <p>
                                                        This section was created with the purpose of anticipating user questions.                                                    </p>
                                                </div>

                                                <div className="accordion-list">
                                                    <ul>
                                                        <li>
                                                            <a data-bs-toggle="collapse" className="collapse" data-bs-target="#accordion-list-1">Does the application work with all types of Braille? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-1" className="collapse show" data-bs-parent=".accordion-list">
                                                                <p>
                                                                    The application will only work with standard Braille Grade 1, without special characters such as diacritics.                                                                    </p>
                                                            </div>
                                                        </li>

                                                        <li>
                                                            <a data-bs-toggle="collapse" data-bs-target="#accordion-list-2" className="collapsed">What types of inputs are allowed? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-2" className="collapse" data-bs-parent=".accordion-list">
                                                                <p >
                                                                    The converted text can be extracted from images, typed from a keyboard, or retrieved from recorded audio in the application. Enabling one method will disable the others.                                                                </p>
                                                            </div>
                                                        </li>

                                                        <li>
                                                            <a data-bs-toggle="collapse" data-bs-target="#accordion-list-3" className="collapsed">Why does it sometimes take longer to display the result? <i className="fa fa-question-circle"></i></a>
                                                            <div id="accordion-list-3" className="collapse" data-bs-parent=".accordion-list">
                                                                <p>
                                                                    Sometimes the final result is displayed later due to the image containing a lot of noise, having numerous closely spaced characters, featuring a hard-to-detect font, or having too many pixels.                                                                </p>
                                                            </div>
                                                        </li>

                                                    </ul>
                                                </div>

                                            </div>

                                            <div className="col-lg-5 align-items-stretch order-1 order-lg-2 img" id="intrebare"></div>
                                        </div>

                                    </div>
                                </section>
                                {/* <!-- End Faq Section --> */}
                                <section id="contact" className="contact">
                                    <div className="container">
                                        <div className="section-title">
                                            <h2 className="h"><i className="fa fa-map-marker" aria-hidden="true"></i> Location</h2>

                                        </div>
                                    </div>

                                    <div>
                                        <div className="gmap_canvas"><iframe width="100%" height="350" id="gmap_canvas" src="https://maps.google.com/maps?q=garii%203a,%20Baia%20mare&t=&z=13&ie=UTF8&iwloc=&output=embed" frameBorder="0" scrolling="no" marginHeight="0" marginWidth="0"></iframe><a href="https://123movies-org.net"></a><br></br>
                                            <a href="https://www.embedgooglemap.net"></a>
                                        </div>
                                    </div>

                                    {/* <!-- End #main --> */}

                                </section>

                            </main>

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
    );
}

export default Home;
