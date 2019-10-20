function changePopup(){
    chrome.browserAction.setPopup({
       popup:"popup.html"
    });
}


var jacob = document.getElementById("jacob");



jacob.onclick = function(){
	location.href = 'popup.html';
};
