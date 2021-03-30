let recommend_button = document.getElementById("recommend-button");
let recommendation = document.getElementById("recommendation")

recommend_button.addEventListener("click", () => {
    let title = document.getElementById("title-input").value;
    let abstract = document.getElementById("abstract-input").value;
    
    while (recommendation.firstChild) {
        recommendation.removeChild(recommendation.firstChild);
    }

    fetch("/recommend", {
        method: "POST",
        headers: {  
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            title: title,
            abstract: abstract
        })
    })
    .then(response => response.json())
    .then(data => {
        for (paper_data of data) {
            paper = document.createElement("div");
            paper.className = "paper";


            paper_title = document.createElement("span");
            paper_title.className = "title"; 
            paper_title.innerHTML = paper_data["title"];

            paper_abstract = document.createElement("p");
            paper_abstract.className = "abstract";
            paper_abstract.innerHTML = paper_data["abstract"];

            paper.appendChild(paper_title);
            paper.appendChild(paper_abstract);

            recommendation.appendChild(paper)
        }
    })
})
