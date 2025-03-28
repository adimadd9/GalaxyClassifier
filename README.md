# Galaxy Classification GUI

This is a university project - a graphical user interface (GUI) for classifying galaxies based on input features. It uses a trained deep learning model to predict the type of galaxy and provides visualization options for analyzing the results.

## Features
- **Manual Classification**: Users can input galaxy features and get a predicted classification.
- **CSV File Processing**: Allows batch classification of multiple galaxies from a CSV file.
- **Data Visualization**: Generates bar charts to display classification results.
- **User-Friendly Interface**: Built with `customtkinter` for a modern and easy-to-use design.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Required Python packages (see below)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
If `requirements.txt` is missing, install the following manually:
```bash
pip install customtkinter tensorflow pandas numpy matplotlib scikit-learn pillow
```

## Running the Application
Run the Python script using:
```bash
python main.py
```

To run the executable (if built using PyInstaller):
```bash
./dist/main.exe
```
Or from the command line:
```bash
cd dist
main.exe
```

## Usage
1. **Manual Classification**:
   - Enter numerical values for galaxy features.
   - Click `Classify` to get a prediction.

2. **Batch Classification**:
   - Click `Load CSV` and select a CSV file with galaxy data.
   - The program will classify each galaxy and save predictions in `data/predictions.csv`.

3. **Visualization**:
   - Click `View Graph` to see the distribution of classifications.

4. **Download Results**:
   - Click `Download Predictions` to save the classification results.

## File Structure
```
project-folder/
│-- data/
│   ├── galaxies.csv           # Example dataset (must be provided)
│   ├── predictions.csv        # The CSV file with the made predictions
│   └── dataDescription.md     # Description of dataset fields
│-- img/
│   ├── bg.jpg                 # Background image
│   └── icon.ico               # Application icon
│-- model/
│   └── galaxy_classifier.h5   # Trained deep learning model
│-- src/
│   ├── main.py                # Main application script
│   ├── preprocess.py          # Data preprocessing functions
│   ├── model.py               # Model build function
│   └── train.py               # Script for training the model
```

## Troubleshooting
- **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`.
- **FileNotFoundError**: Ensure that `galaxies.csv` and `galaxy_classifier.h5` are in their respective directories.
- **GUI Not Opening**: Verify that `customtkinter` is installed correctly.

## Building the Executable
To create an executable using PyInstaller:
```bash
pyinstaller --onefile --windowed --add-data "model/galaxy_classifier.h5;model" --add-data "data/*;data" --add-data "img/*;img" --icon="img/icon.ico" main.py
```

### Data Source

This project uses data from the **Galaxy Zoo** project, a citizen science initiative where volunteers classify galaxies. The dataset used in this project is publicly available for research and academic purposes. You can find more information about Galaxy Zoo and access the dataset [here](https://data.galaxyzoo.org).

## License
This project is released under the MIT License. - see the [LICENSE](./LICENSE) file for details.

## Author
Developed by **Doncilă Denis**.

---
Enjoy using the **Galaxy Classification GUI**!

