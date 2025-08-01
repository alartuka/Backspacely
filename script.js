const form = document.getElementById('code-form');
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const repoUrl = form.repoUrl.value;
    const prompt = form.prompt.value;
    const codeRequest = new URLSearchParams();
    codeRequest.append('repoUrl', repoUrl);
    codeRequest.append('prompt', prompt);
    fetch('/code', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: codeRequest
    })
    .then(response => response.json())
    .then((data) => {
        console.log(data);
        // display the generated code here
    })
    .catch((error) => {
        console.error(error);
    });
});