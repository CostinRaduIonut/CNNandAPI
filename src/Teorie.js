import React, { useContext } from 'react';
import { LanguageContext } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
import './Home.css';
import './Include.css'
import './Teorie.css'
import { useNavigate } from 'react-router-dom';

const Teorie = () => {
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
                                    <a href="#">Teorie</a>
                                </li>
                                <li className="breadcrumb-item active">Alfabetul Braille</li>
                            </ol> */}

                            <div className="container1">
                                <div className="row">
                                    <div className="column-66">
                                        <h1 id="unu" className="xlarge-font"><b>Alfabetul Braille </b></h1>
                                        <p>Toate semnele și literele se formează prin combinații de puncte în cadrul unui grup de 6 puncte.
                                            Fiecare literă, de la A la Z are câte un corespondent în alfabetul Braille, compus din puncte ieșite în relief.</p>

                                    </div>
                                    <div className="column-33">
                                        <img src={"../alfa.jpg"} width="335" height="471"></img>
                                    </div>
                                </div>
                            </div>

                            <div className="container1" style={{ backgroundColor: "#83A697" }}>
                                <div className="row">
                                    <div className="column-33">
                                        <img src={"../cifre.jpg"} alt="App" width="335" height="471"></img>
                                    </div>
                                    <div className="column-66">
                                        <h1 id="doi" className="xlarge-font"><b style={{color:"white"}}>Cifre și litere mari în Braille</b></h1>
                                        <p style={{color:"white"}}>Cifrele sunt formate din primele 10 litere (0-9) ale alfabetului, cu semnul de cifră în față, iar literele mari sunt marcate de prezența simbolului de majusculă în față</p>
                                    </div>
                                </div>
                            </div>


                            <div className="container1">
                                <div className="row">
                                    <div className="column-66">
                                        <h1 id="trei" className="xlarge-font"><b>Semne de punctuație</b></h1>
                                        <p></p>
                                    </div>
                                    <div className="column-33">
                                        <img src="../punctuatie_ro.jpg" width="335" height="471"></img>
                                    </div>
                                </div>
                            </div>
                            <div className="container1" style={{ backgroundColor: "#83A697" }}>
                                <div className="row">
                                    <div className="column-33">
                                        <img src={'../Louis.jpg'} alt="App" width="335" height="471"></img>
                                    </div>
                                    <div className="column-66">
                                        <h1 id="patru" className="xlarge-font"><b style={{color:"white"}}>Istoric</b></h1>
                                        <p style={{color:"white"}}>Louis Braille s-a născut la Coupvray, localitate situată la 40 de kilometri de Paris.
                                            Şi-a pierdut vederea la vârsta de trei ani, în urma unui accident, Braille a studiat la Institutul Naţional pentru Tinerii Nevăzători din Paris,
                                            dar cea mai mare parte a timpului liber şi-o dedica încercărilor de a găsi o modalitate eficientă de a reprezenta, tactil, literele şi numerele.
                                            Louis Braille a aflat, însă, despre  ”scrierea pe timp de noapte”, un sistem de comunicare prin puncte embosate. Louis a fost foarte impresionat
                                            şi şi-a petrecut următorii trei ani încercând să dezvolte proiectul. Aşa a apărut alfabetul cu puncte în basorelief, folosit şi în prezent de nevăzători</p>
                                    </div>
                                </div>
                            </div>


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
                            {/* <ol className="breadcrumb">
                                <li className="breadcrumb-item">
                                    <a href="#">Theory</a>
                                </li>
                                <li className="breadcrumb-item active">Theory</li>
                            </ol> */}
                            <div className="container1">
                                <div className="row">
                                    <div className="column-66">
                                        <h1 id="unu" className="xlarge-font"><b>The Braille alphabet</b></h1>
                                        <p>All the signs and letters are formed through combinations of dots within a group of 6 dots.
                                            Each letter, from A to Z, has a corresponding representation in the Braille alphabet, composed of raised dots.</p>

                                    </div>
                                    <div className="column-33">
                                        <img src={"../alfa.jpg"} width="335" height="471"></img>
                                    </div>
                                </div>
                            </div>

                            <div className="container1" style={{ backgroundColor: "#83A697" }}>
                                <div className="row">
                                    <div className="column-33">
                                        <img src={"../cifre_en.jpg"} alt="App" width="335" height="471"></img>
                                    </div>
                                    <div className="column-66">
                                        <h1 id="doi" className="xlarge-font"><b style={{color:"white"}}>Numbers and capital letters in Braille.</b></h1>
                                        <p style={{color:"white"}}>Numbers are formed by the first 10 letters (0-9) of the alphabet, with a numeral sign in front, while capital letters are indicated by the presence of a capitalization symbol in front.</p>                                    </div>
                                </div>
                            </div>


                            <div className="container1">
                                <div className="row">
                                    <div className="column-66">
                                        <h1 id="trei" className="xlarge-font"><b>Punctuation marks</b></h1>
                                        <p></p>
                                    </div>
                                    <div className="column-33">
                                        <img src="../punctuatie.jpg" width="335" height="471"></img>
                                    </div>
                                </div>
                            </div>
                            <div className="container1" style={{ backgroundColor: "#83A697" }}>
                                <div className="row">
                                    <div className="column-33">
                                        <img src={'../Louis.jpg'} alt="App" width="335" height="471"></img>
                                    </div>
                                    <div className="column-66">
                                        <h1 id="patru" className="xlarge-font"><b style={{color:"white"}}>History</b></h1>
                                        <p style={{color:"white"}}>Louis Braille was born in Coupvray, a town located 40 kilometers from Paris.
                                            He lost his sight at the age of three following an accident. Braille studied
                                            at the National Institute for Blind Youth in Paris, but he dedicated most of
                                            his free time to finding an efficient way to represent letters and numbers
                                            tactually. However, Louis Braille came across "night writing," a system of
                                            communication using embossed dots. He was greatly impressed and spent the
                                            next three years working on developing the project. This led to the creation
                                            of the raised dot alphabet, which is still used by the blind today.</p>
                                    </div>
                                </div>
                            </div>


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
                </div></>)}
        </>
    )
}
export default Teorie