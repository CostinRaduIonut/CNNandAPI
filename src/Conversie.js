import React, { useState, useEffect, useRef, useContext } from 'react';
import { LanguageContext } from './LanguageContext';
import LanguageSelector from './LanguageSelector';
import axios from 'axios';
import './Home.css';
import './Include.css'
import './Conversie.css'
import { toast } from "react-toastify"

import { useLocation, useNavigate } from 'react-router-dom';

const Conversie = () => {
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
    const toPastrate = () => {
        navigate("/Pastrate")
    }
    const toTermeni = () => {
        navigate("/Termeni")
    }
    //get return type
    const [selectedOption, setSelectedOption] = useState('image');

    function handleSelectChange(event) {
        setSelectedOption(event.target.value);
    }
    //send img
    const [file, setFile] = useState(null);
    const [check_spelling, setCheck_spelling] = useState(false);
    const [keep, setKeep] = useState(false);
    const myParagraphRef = useRef('');
    const [image, setImage] = useState(null);

    const handleSubmit = async (event) => {
        newUser()

        const text = myParagraphRef.current.textContent;
        if (text.length <= 1 && file === null) {
            if (selectedLanguage === "română")
                toast.error("Nu ați introdus nimic")
            else
                toast.error("No input inserted")
        }

        else {
            event.preventDefault();
            const formData = new FormData();
            formData.append('file', file);
            formData.append('check_spelling', check_spelling);
            formData.append('file_type', selectedOption);

            const formData2 = new FormData();
            formData2.append('text', text);
            formData2.append('file_type', selectedOption);
            formData2.append('check_spelling', check_spelling);



            try {
                if (inputValues.input1.trim().length > 0 || transcript.length > 0) {
                    const response = await axios.post('http://localhost:8000/string-to-braille', formData2, { responseType: 'blob' }, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        },
                    });
                    downlandResponse(response)
                    if (keep == true) {
                        var adresaIP
                        axios.get("https://api.ipify.org?format=json").then((r) => {
                            adresaIP = r.data.ip.toString();

                            const formData3 = new FormData();
                            formData3.append("ip", adresaIP);
                            formData3.append('data', text);


                            axios.post('http://localhost:8000/put_kept_string', formData3, { responseType: 'blob' }, {
                                headers: {
                                    'Content-Type': 'multipart/form-data'
                                },
                            });
                        })
                    }
                }
                else {
                    const response = await axios.post('http://localhost:8000/text-to-braille', formData, { responseType: 'blob' }, {
                        headers: {
                            'Content-Type': 'multipart/form-data'
                        }
                    });
                    downlandResponse(response)
                }

            } catch (error) {
                console.error(error);
                toast.error(error)
            }
        }
    };

    function downlandResponse(response) {
        if (selectedOption == 'image')
            downlandImg(response)
        else
            downlandAudio(response)
    }
    //descarcare img
    function downlandImg(response) {
        // console.log(response.data);
        const url = window.URL.createObjectURL(new Blob([response.data]));
        setImage(url);

        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'image.jpg');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    //descarcare audio
    function downlandAudio(response) {
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', 'audio.mp3');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    function handleFileChange(event) {
        setFile(event.target.files[0]);
    };

    const handlecheck_spellingChange = (event) => {
        if (check_spelling == false)
            if (selectedLanguage === 'română')
                toast.info("Autocorectorul funcționează doar în limba engleză")
            else
                toast.info("The spell checker works only in English.")

        setCheck_spelling(event.target.checked);
    };
    const handlekeep_keepChange = (event) => {
        if (inputValues.input1.trim().length > 0 || transcript.length > 0)
            setKeep(event.target.checked)
        else
            setKeep(false)



    }
    //identificare ip
    var ipAddress
    function newUser() {
        axios.get("https://api.ipify.org?format=json").then((response) => {
            ipAddress = response.data.ip;
            // do something with the IP address
            // console.log(ipAddress)
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
            fetch("http://localhost:8000/new_user", requestOptions)
                .then((response) => response.json())
                .then((text) => {
                    // console.log(text);
                    localStorage.setItem("unique", text.unique_count)
                })
        });
    }

    // pentru speech to text

    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();


    const [transcript, setTranscript] = useState('');
    const [isListening, setIsListening] = useState(false);

    useEffect(() => {
        recognition.onresult = event => {
            const current = event.resultIndex;
            const newTranscript = event.results[current][0].transcript;
            setTranscript(transcript + " " + newTranscript);
        };
    }, [transcript]);

    const startRecognition = () => {
        if (!isListening) {
            recognition.start();
            setIsListening(true);
        }
    };

    const stopRecognition = () => {
        recognition.stop();
        setIsListening(false);
    };
    const clear = () => {
        setTranscript('')
        window.location.reload()
    }


    //dezactivare alte inputuri
    const [inputValues, setInputValues] = useState({
        input1: '',
        input2: '',
        input3: '',
    });


    function handleInputChange(event) {
        const { name, value } = event.target;
        setInputValues(prevValues => ({
            ...prevValues,
            [name]: value
        }));

        // if (inputValues.input2.trim().length > 0) {
        //     handleFileChange()
        // }

    }

    const isDisabled = Object.values(inputValues).some(value => value.trim().length > 0 || transcript.length > 0);

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
                            {/* <!-- Breadcrumbs--> */}
                            {/* <ol className="breadcrumb">
                                <li className="breadcrumb-item">
                                    <a href="#">Conversie</a>
                                </li>
                                <li className="breadcrumb-item active">Conversie</li>
                            </ol> */}


                            <div className="container">
                                <div className="container" >
                                    <div className="shade">
                                        <div className="blackboard">
                                            <div className="form">
                                                <p>
                                                    <label>Text: </label>
                                                    <input type="text" name="input1"
                                                        value={inputValues.input1}
                                                        onChange={handleInputChange} />
                                                </p>
                                                <p>
                                                    <label>Imagine: </label>
                                                    <input type="file"
                                                        id="file"
                                                        name="input2"
                                                        // value={inputValues.input2}
                                                        onChange={handleFileChange}
                                                        //onChange={handleInputChange}
                                                        onClick={(e) => (e.target.value = null)}
                                                        disabled={isDisabled}
                                                        accept="image/jpeg, image/jpg"
                                                    />

                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Autocorector</label> <input type="checkbox" id="check_spelling" checked={check_spelling} onChange={handlecheck_spellingChange}></input>
                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Selectați outputul dorit:</label>
                                                    <select value={selectedOption} onChange={handleSelectChange}>
                                                        <option value="image">Imagine</option>
                                                        <option value="sound">Audio</option>
                                                    </select>
                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Păstrați textul</label><input type="checkbox" id="keep" checked={keep} onChange={handlekeep_keepChange}></input>
                                                    <br></br>
                                                    <button onClick={toPastrate}>Vedeți rezultatele păstrate</button>
                                                </p>
                                                <p>
                                                    <label>Pentru mesaj vocal, alegeti optiunile de mai jos: </label>
                                                </p>
                                                <p className="wipeout">
                                                    <button onClick={startRecognition}>Start<i style={{ paddingLeft: "0.5%" }} className="fa fa-microphone" aria-hidden="true"></i></button>
                                                    <button onClick={stopRecognition}>Stop<i style={{ paddingLeft: "0.5%" }} className="fa fa-microphone-slash" aria-hidden="true"></i></button>
                                                    <button onClick={clear}>Curăță<i style={{ paddingLeft: "0.5%" }} className="fa fa-refresh" aria-hidden="true"></i></button>
                                                    <br></br>
                                                    <p ref={myParagraphRef} >{transcript + " " + inputValues.input1}</p>
                                                    <hr></hr>
                                                    <input type="submit" value="Convertește și descarcă" onClick={handleSubmit} />
                                                </p>
                                            </div>

                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <footer className="sticky-footer" style={{
                        position: "fixed",
                        bottom: "0",
                    }}>
                        <div className="container">
                            <div className="text-center">
                                ©Costin Radu Ionuț <a href="https://www.facebook.com/ionut.radu.5036/" className="fa fa-facebook"></a><a href="https://www.instagram.com/" className="fa fa-instagram"></a>
                                <a onClick={toTermeni} style={{ color: "black", float: "right" }}>Termenii și condițiile</a>
                            </div>
                        </div>
                    </footer>
                </div >
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
                                    <a href="#">Conversion</a>
                                </li>
                                <li className="breadcrumb-item active">Conversion</li>
                            </ol> */}
                            <div className="container">
                                <div className="container" >
                                    <div className="shade">
                                        <div className="blackboard">
                                            <div className="form">
                                                <p>
                                                    <label>Text: </label>
                                                    <input type="text" name="input1"
                                                        value={inputValues.input1}
                                                        onChange={handleInputChange} />
                                                </p>
                                                <p>
                                                    <label>Image: </label>
                                                    <input type="file"
                                                        id="file"
                                                        name="input2"
                                                        // value={inputValues.input2}
                                                        onChange={handleFileChange}
                                                        //onChange={handleInputChange}
                                                        onClick={(e) => (e.target.value = null)}
                                                        disabled={isDisabled}
                                                        accept="image/jpeg, image/jpg"
                                                    />

                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Autocorrect</label> <input type="checkbox" id="check_spelling" checked={check_spelling} onChange={handlecheck_spellingChange}></input>
                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Select the desired output:</label>
                                                    <select value={selectedOption} onChange={handleSelectChange}>
                                                        <option value="image">Image</option>
                                                        <option value="sound">Sound</option>
                                                    </select>
                                                    <br></br>
                                                    <label style={{ paddingRight: "1%" }}>Keep the text</label><input type="checkbox" id="keep" checked={keep} onChange={handlekeep_keepChange}></input>
                                                    <br></br>
                                                    <button onClick={toPastrate}>View saved results</button>
                                                </p>
                                                <p>
                                                    <label>For voicemail, choose the options below: </label>
                                                </p>
                                                <p className="wipeout">
                                                    <button onClick={startRecognition}>Start<i style={{ paddingLeft: "0.5%" }} className="fa fa-microphone" aria-hidden="true"></i></button>
                                                    <button onClick={stopRecognition}>Stop<i style={{ paddingLeft: "0.5%" }} className="fa fa-microphone-slash" aria-hidden="true"></i></button>
                                                    <button onClick={clear}>Clear<i style={{ paddingLeft: "0.5%" }} className="fa fa-refresh" aria-hidden="true"></i></button>
                                                    <br></br>
                                                    <p ref={myParagraphRef} >{transcript + " " + inputValues.input1}</p>
                                                    <hr></hr>
                                                    <input type="submit" value="Convert and downland" onClick={handleSubmit} />
                                                </p>
                                            </div>

                                        </div>
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
                </div>
            </>)}
        </>


    )
}
export default Conversie
