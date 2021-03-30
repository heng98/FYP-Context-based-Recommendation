from flask import (
    Flask,
    request,
    render_template,
    jsonify
)

from api.corpus import Corpus
from api.document_embedding import DocumentEmbeddingModel
from Transformer.candidate_selector.ann.ann_annoy import ANNAnnoy
from api.candidate_selection.ann_candidate_selector import ANNCandidateSelector



app = Flask(__name__)

corpus = Corpus()
document_embedding_model = DocumentEmbeddingModel("weights_3")

ann = ANNAnnoy.load("dblp.ann")
candidate_selector = ANNCandidateSelector(ann, 8, corpus)


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route('/recommend', methods=["POST"])
def recommend():
    if request.method == "POST":
        title = request.json["title"]
        abstract = request.json["abstract"]
        
        query_embedding = document_embedding_model.embed(title, abstract)
        candidate_papers = candidate_selector.get_candidate(query_embedding)
        result = [
            {
                "title": c["title"],
                "abstract": c["abstract"]
            }
            for c, _ in candidate_papers
        ]

        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)




