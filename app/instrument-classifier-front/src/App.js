import DragDrop from './DragDrop';
import './index.css';
import './stil.css';

function App() {
  return (
    <div className="App">
      <h1 className="App-title">Instrument classificator brought to you by RiTehc</h1>
      <div className="App-body">
        <div className='wrapie'><p className='paragraph3'>Hello, dear user!</p>
        <p className='paragraph3'>In the section below you can drag and drop wav file or click and select it from your local machine.</p>
        <p className='paragraph3'>
        Our application will do some serious and really complex AI stuff on your file and return instruments which can be found playing.</p>
        <p className='paragraph3'>You also have separate 3 second windows so you can see what is playing when. Windows overlap, so don't worry if there are more windows than you expected.</p></div>
        <DragDrop ></DragDrop>
      </div>
    </div>
  );
}

export default App;