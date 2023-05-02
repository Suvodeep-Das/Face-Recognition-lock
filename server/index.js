import "@tensorflow/tfjs-node";
import express from "express";
import bodyParser from "body-parser";
import fileUpload from "express-fileupload";
import fs from "fs";
import * as faceapi from "@vladmandic/face-api";
import { Canvas, Image, ImageData } from "canvas";
import * as canvas from "canvas";
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

const REF_COUNT = 14;
const descriptionsMap = {};

async function LoadModels() {
  // Load the models
  // __dirname gives the root directory of the server
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
}

async function GenDescriptor(_label) {
  try {
    let counter = 0;
    const descriptions = [];

    for (let i = 1; i <= REF_COUNT; i++) {
      const img = await canvas.loadImage(`./reference/${_label}/${i}.jpg`);
      counter = (i / REF_COUNT) * 100;
      console.log(`Progress = ${counter}%`);
      // Read each face and save the face descriptions in the descriptions array
      const detections = await faceapi
        .detectSingleFace(img)
        .withFaceLandmarks()
        .withFaceDescriptor();
      descriptions.push(detections.descriptor);
      descriptionsMap[_label] = descriptions;

      //console.log(descriptionsMap);
    }
  } catch (error) {
    console.error(error);
  }
}

async function ValidateImage(image) {
  // Get all the face data and loop through each of them to read the data
  let faces = [];
  Object.entries(descriptionsMap).map(([_label, _descriptions], _index) => {
    for (let j = 0; j < _descriptions.length; j++) {
      _descriptions[j] = new Float32Array(Object.values(_descriptions[j]));
    }

    faces[_index] = new faceapi.LabeledFaceDescriptors(_label, _descriptions);
  });

  // Load face matcher to find the matching face
  const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);

  // Read the image using canvas or other method
  const img = await canvas.loadImage(image);
  let temp = faceapi.createCanvasFromMedia(img);
  // Process the image for the model
  const displaySize = { width: img.width, height: img.height };
  faceapi.matchDimensions(temp, displaySize);

  // Find matching faces
  const detections = await faceapi
    .detectAllFaces(img)
    .withFaceLandmarks()
    .withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((d) =>
    faceMatcher.findBestMatch(d.descriptor)
  );
  return results;
}

function main() {
  LoadModels()
    .then(async () => {
      await GenDescriptor("saikat");
    })
    .then(async () => {
      //const result = await ValidateImage("./reference/saikat/6.jpg");
      //console.log(result);
    })
    .then(() => {
      const server = express();
      server.use(
        fileUpload({
          /* useTempFiles: true, */
        })
      );

      server.get("/health", async (req, res) => {
        res.status(200).end("OK");
      });

      server.post("/validate", async (req, res) => {
        try {
          //console.log(req.files.imageFile);
          //fs.writeFileSync(`./abc${counter++}.jpg`, req.files.imageFile.data);

          const result = await ValidateImage(req.files.imageFile.data);
          console.log(JSON.stringify(result));

          if (!result.length || result[0]._label === "unknown")
            res.status(200).end("LOCK");
          else res.status(200).end("UNLOCK");
        } catch (error) {
          console.error(error);
          res.status(200).end("LOCK");
        }
      });

      server.listen(4000, () => {
        console.log("Server Online!");
      });
    });
}

main();
