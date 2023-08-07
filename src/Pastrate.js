import React, { useState, useEffect, useRef, useContext } from 'react';
import { LanguageContext } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';
import './Home.css';
import './Include.css'
import './Pastrate.css'
const Pastrate = () => {
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
    const [data, setData] = useState([]);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        try {
            const response = await axios.get('https://api.ipify.org?format=json');
            const ipAddress = response.data.ip;

            const requestOptions = {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json",
                },
                body: JSON.stringify({
                    ipAddress: ipAddress
                }),
            };

            fetch("http://localhost:8000/get_kept_string", requestOptions)
                .then((res) => res.json())
                .then((data) => setData(data.data))

        } catch (error) {
            console.error(error);
        }
    };

    const deleteAll = async () => {
        const response = await axios.get('https://api.ipify.org?format=json');
        const ipAddress = response.data.ip;

        const requestOptions = {
            method: "DELETE",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({
                ipAddress: ipAddress
            }),
        };

        fetch("http://localhost:8000/delete_all", requestOptions)
        window.location.reload()
    };

    const deleteOne = async (data) => {
        const requestOptions = {
            method: "DELETE",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
            },
            body: JSON.stringify({
                data: data
            }),
        };

        fetch("http://localhost:8000/delete_one", requestOptions)
        window.location.reload()
    };

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
                    <div>
                        <div className="content-wrapper">
                            <div className="container-fluid" style={{padding:"3%"}}>
                                {/* <!-- Breadcrumbs--> */}
                                {/* <ol className="breadcrumb">
                                    <li className="breadcrumb-item">
                                        <a href="#">Date pastrate</a>
                                    </li>
                                    <li className="breadcrumb-item active">Date pastrate</li>
                                </ol> */}
                                <h1>Datele pe care ați ales să le păstrați</h1>

                                {data.length !== 0 ?
                                    (<>
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Data</th>
                                                    <th> <button onClick={deleteAll}>Șterge toate</button></th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {data.map((item, index) => (
                                                    <tr key={index}>
                                                        <td>{item}</td>
                                                        <td><button onClick={() => deleteOne(item)}>Șterge</button></td>
                                                    </tr>

                                                ))}
                                            </tbody>
                                        </table>
                                    </>) : (<>
                                        <table>
                                            <thead>
                                                <tr>
                                                    <th>Nu ați salvat nimic</th>
                                                </tr>
                                            </thead>
                                        </table>
                                    </>)}
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
                        <div className="container-fluid" style={{padding:"3%"}}>
                            {/* <ol className="breadcrumb">
                                <li className="breadcrumb-item">
                                    <a href="#">Data stored</a>
                                </li>
                                <li className="breadcrumb-item active">Data stored</li>
                            </ol> */}
                            <h1>The data you have chosen to store.</h1>

                            {data.length !== 0 ?
                                (<>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>Data</th>
                                                <th> <button onClick={deleteAll}>Delete all</button></th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {data.map((item, index) => (
                                                <tr key={index}>
                                                    <td>{item}</td>
                                                    <td><button onClick={() => deleteOne(item)}>Delete</button></td>
                                                </tr>

                                            ))}
                                        </tbody>
                                    </table>
                                </>) : (<>
                                    <table>
                                        <thead>
                                            <tr>
                                                <th>You haven't saved anything.</th>
                                            </tr>
                                        </thead>
                                    </table>
                                </>)}

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
};

export default Pastrate