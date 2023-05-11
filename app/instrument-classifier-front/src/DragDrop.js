import React, { useState } from "react";
import { FileUploader } from "react-drag-drop-files";
import BarChart from './BarChart.js'
import './stil.css'

const fileTypes = ["WAV"];

function DragDrop() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [windowData, setWindowData] = useState([])

  const handleChange = (file) => {
    setFile(file);
  };

  const onDrop = (file) => {
    sendRequest(file);
  };

  const onSelect = (file) => {
    sendRequest(file)
  };

  const sendRequest = (file) => {

    var formdata = new FormData();
    formdata.append("file", file, file.name)

    const requestOptions = {
      method: 'POST',
      body: formdata
    };

    fetch("http://127.0.0.1:8000/window_predictions", requestOptions)
      .then(response => response.text())
      .then(result => {
        let arr = JSON.parse(result);
        let windowChartData = []
        arr.map((obj) => windowChartData.push({
          labels: Object.keys(obj),
          datasets: [
            {
              label: "Predicted",
              data: Object.values(obj),
              backgroundColor: ["rgba(81, 226, 245, 0.9)"],
            }
          ]
        }
        )
        )
        setWindowData(windowChartData)
      }
      )
      .catch(error => console.log('error', error));

    fetch("http://127.0.0.1:8000/final_prediction", requestOptions)
      .then(response => response.text())
      .then(result => {
        let obj = JSON.parse(result);
        let keys = Object.keys(obj);
        let values = Object.values(obj);
        setData({
          labels: keys,
          datasets: [
            {
              label: "Predicted",
              data: values,
              backgroundColor: ['rgba(81, 226, 245, 0.9)']
            }
          ]
        }
        )
      }
      )
      .catch(error => console.log('error', error));
  }
  return (
    <div className="wrap-all">
      <FileUploader className="file-div"
        style={{display: "flex", alignItems: "center", justifyContent: "center"}}
        onDrop={onDrop}
        onSelect={onSelect}
        maxSize="10"
        minSize="0.1"
        handleChange={handleChange} name="file" types={fileTypes} />
      {data && <div><p className="paragraph1">Uploaded {file.name}</p><div style={{ width: 500 }} className="final-container"><p className="paragraph1">Final prediction</p><BarChart chartData={data} /></div></div>}
      <div className="window-container">{windowData.map((data, index) => (
        <div className="chart-window" style={{ width: 350 }}><p className="paragraph2">Window {index + 1}</p><BarChart chartData={data} /></div>
      ))}</div>
    </div>
  );
}

export default DragDrop;