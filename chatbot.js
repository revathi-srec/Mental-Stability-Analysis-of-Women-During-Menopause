function sendMessage() {

let input = document.getElementById("userInput").value;

if(input.trim() === "") return;

let chatbox = document.getElementById("chatbox");

// show user message
chatbox.innerHTML += "<div class='user-msg'><b>You:</b> " + input + "</div>";

fetch("/ai_chat", {
method: "POST",
headers: {
"Content-Type": "application/json"
},
body: JSON.stringify({message: input})
})

.then(response => response.json())
.then(data => {

chatbox.innerHTML += "<div class='bot-msg'><b>AI:</b> " + data.reply + "</div>";

chatbox.scrollTop = chatbox.scrollHeight;

});

document.getElementById("userInput").value = "";

}