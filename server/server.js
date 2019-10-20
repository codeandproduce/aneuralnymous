var express = require('express');
var app = express();
var path = require('path');

const fs = require('fs');

const PORT = process.env.PORT || 3000;


var count = 0;

const multer = require("multer");

const handleError = (err, res) => {
  res
    .status(500)
    .contentType("text/plain")
    .end("Oops! Something went wrong!");
};

const upload = multer({
  dest: path.join(__dirname, "./tmp/")
  // you might also want to set some limits: https://github.com/expressjs/multer#limits
});

app.use(express.static(path.join(__dirname, './uploads')));


app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*"); // update to match the domain you will make the request from
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next();
});

app.get('/whatisadversary', (req, res) => {
	res.sendFile(path.join(__dirname, './index.html'));
});


app.post("/upload", upload.single("picture-file"), (req, res) => {
  const tempPath = req.file.path;
  const targetPath = path.join(__dirname, `./uploads/image${count}.png`);	
  
  fs.rename(tempPath, targetPath, err => {
  	if(err){
      console.log(err)
      return handleError(err, res);
    }
    count = count + 1;

    create_adversary(res, targetPath);
  });
});


function create_adversary(res, img_name){
  console.log("got image")
  console.log(img_name)
  
  var spawn = require("child_process").spawn; 
  var process = spawn('python3',["./model/image_convert.py", img_name]); 
  
  process.stdout.on('data', (data) => { 
    console.log(data)
    var new_image_file_name = data.toString().replace('\n','');
    var real_path = path.join(__dirname, './adversary/' + new_image_file_name);

    var img = fs.readFile(real_path, (err, data) => {
      var contentType = 'image/png';
      var base64 = Buffer.from(data).toString('base64');
      base64='data:image/png;base64,'+base64;
      res.send(base64);
    });



    // res.status(202).sendFile(real_path);

  }); 

}




app.listen(PORT, () => {
	console.log(`Server up and running on ${PORT}`);
});
