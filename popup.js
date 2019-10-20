// Copyright 2018 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

'use strict';


let dragZone = document.getElementById('dropzone');
let displayZone = document.getElementById('display-zone');
let takeZone = document.getElementById('takezone');
let displayZone2 = document.getElementById('display-zone-2');

// chrome.storage.sync.get('color', function(data) {
//   changeColor.style.backgroundColor = data.color;
//   changeColor.setAttribute('value', data.color);
// });

// changeColor.onclick = function(element) {
//   let color = element.target.value;
//   chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
//     chrome.tabs.executeScript(
//         tabs[0].ißd,
//         {code: 'document.body.style.backgroundColor = "' + color + '";'});
//   });
// };

dragZone.onclick = function(element){
	console.log(click);
}

dragZone.addEventListener("dragover", function( event ) {
  // prevent default to allow drop
  event.preventDefault();
  console.log("DRAGOVER");
});


dragZone.addEventListener('drop', function(e) {
    e.stopPropagation();
    e.preventDefault();
    var files = e.dataTransfer.files; // Array of all files

    for (var i=0, file; file=files[i]; i++) {
        if (file.type.match(/image.*/)) {
        	var reader = new FileReader();

            reader.onload = (e2) => {

            	uploadImage(e2.target.result);

                // finished reading file data.
              $("#display-zone").css("display", "block");
              $("#dropzone").css("display", "none");
              var img = document.createElement('img');
              img.src= e2.target.result;
              img.className="dropped-img loading";
              displayZone.appendChild(img);
            }

            reader.readAsDataURL(file); 
        }
    }
});


const b64toBlob = (b64Data, contentType='', sliceSize=512) => {
  const byteCharacters = atob(b64Data);
  const byteArrays = [];

  for (let offset = 0; offset < byteCharacters.length; offset += sliceSize) {
    const slice = byteCharacters.slice(offset, offset + sliceSize);

    const byteNumbers = new Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      byteNumbers[i] = slice.charCodeAt(i);
    }

    const byteArray = new Uint8Array(byteNumbers);
    byteArrays.push(byteArray);
  }

  const blob = new Blob(byteArrays, {type: contentType});
  return blob;
}




function uploadImage(img){

	var url = 'http://34.237.243.73:3000/upload';

	var byteCharacters = atob(img.replace(/^data:image\/(png|jpeg|jpg);base64,/, ''));
	var byteNumbers = new Array(byteCharacters.length);
	for (var i = 0; i < byteCharacters.length; i++) {
	  byteNumbers[i] = byteCharacters.charCodeAt(i);
	}

	var byteArray = new Uint8Array(byteNumbers);
	var blob = new Blob([ byteArray ], {
	   type : undefined
	});

	var formData = new FormData();
	formData.append('picture-file', blob);

	$.ajax({
	    url: url, 
	    type: "POST", 
	    cache: false,
      crossDomain: true,
	    contentType: false,
	    processData: false,
	    data: formData
	}).done((result) => {

      $('.loading').removeClass('loading');
      $('.contain-loader').css('display', 'none');
      $('.ad-text').css('display', 'block');
      $('.what-class').css('display', 'block');

      $("#display-zone-2").css("display", "block");
      $("#takezone").css("display", "none");

      var img = document.createElement('img');
      img.src= result;
      img.className="dropped-img";
      displayZone2.appendChild(img);
  







  }).catch((err) => {
    console.log(err);
  });

}


	// var data = new FormData();
 //    data.append('file', $('#file')[0].files[0]);



 //    $.ajax({
 //        type:'POST',
 //        url: $(this).attr('action'),
 //        data:formData,
 //        cache:false,
 //        contentType: false,
 //        processData: false,
 //        success:function(data){
 //            console.log("success");
 //            console.log(data);
 //        },
 //        error: function(data){
 //            console.log("error");
 //            console.log(data);
 //        }
 //    });

