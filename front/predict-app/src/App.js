import React from "react";
import CardContent from "@material-ui/core/CardContent";
import Table from "@material-ui/core/Table";
import TableBody from "@material-ui/core/TableBody";
import TableCell from "@material-ui/core/TableCell";
import TableHead from "@material-ui/core/TableHead";
import TableRow from "@material-ui/core/TableRow";
import CardActionArea from "@material-ui/core/CardActionArea";
import CardActions from "@material-ui/core/CardActions";
import CardMedia from "@material-ui/core/CardMedia";
import { useState } from "react";
import Card from "@material-ui/core/Card";

function App({ classes }) {
  const [image, setImage] = useState("");
  const [imageUrl, setImageUrl] = useState("");

  const handleImageUpload = event => {
    event.preventDefault();
    const reader = new FileReader();
    const file = event.target.files[0];

    reader.onloadend = () => {
      setImage(file);
      setImageUrl(reader.result);
    };

    reader.readAsDataURL(file);
  };

  return (
    <div className="App">
      <Card>
        <CardActions>
          <input type="file" onChange={e => handleImageUpload(e)} />
          <label for="file">Upload DICOM image</label>
        </CardActions>
        <CardActionArea>
          <CardMedia image={imageUrl}></CardMedia>
          <CardContent>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Confidence</TableCell>
                  <TableCell>X</TableCell>
                  <TableCell>Y</TableCell>
                  <TableCell>W</TableCell>
                  <TableCell>H</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>...</TableCell>
                  <TableCell>...</TableCell>
                  <TableCell>...</TableCell>
                  <TableCell>...</TableCell>
                  <TableCell>...</TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </CardContent>
        </CardActionArea>
      </Card>
    </div>
  );
}

export default App;
