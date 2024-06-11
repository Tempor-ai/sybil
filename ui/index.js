import React, {useState} from "react";
import Grid from "@material-ui/core/Grid";
import Typography from "@material-ui/core/Typography";
import Button from "@material-ui/core/Button";

import FileUploader from "../../common/FileUploader";

function Sybil () {

    const [uploadedFiles, setUploadedFiles] = useState({})

    const handleCsvFileUpload = () => {
        return;
    }

    const handleJsonFileUpload = () => {
        return;
    }

    const validateFile = () => {
        return;
    }

    return (
        <>
            <Typography variant="body2" component="p" align="left" style={{ marginBottom: 20, marginTop: 20 }}>
                Select CSV file to upload
            </Typography>
            <FileUploader
                name="csvURL"
                type="file"
                uploadedFiles={uploadedFiles['csvFile']}
                handleFileUpload={handleCsvFileUpload}
                setValidationStatus={validateFile}
                maxFileNames={1}
                fileAccept=".csv"
            />
            <Typography variant="body2" component="p" align="left" style={{ marginBottom: 20, marginTop: 20 }}>
                Select JSON file to upload
            </Typography>
            <FileUploader
                name="jsonURL"
                type="file"
                uploadedFiles={uploadedFiles['jsonFile']}
                handleFileUpload={handleJsonFileUpload}
                setValidationStatus={validateFile}
                maxFileNames={1}
                fileAccept=".json"
            />
        </>
    )
}

export default Sybil