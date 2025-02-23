document.getElementById('summarizeBtn').addEventListener('click', function() {
    let text = document.getElementById('inputText').value;
    
    if (text.trim() === "") {
        alert("Please enter some text.");
        return;
    }

    let words = text.split(" ");
    let summary = words.slice(0, Math.min(words.length, 20)).join(" ") + "...";
    
    alert("Summary: " + summary);
});
