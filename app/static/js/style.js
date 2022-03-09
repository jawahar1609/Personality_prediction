$(document).ready(function() {
    $('.progress .progress-bar').css("width",
        function() {
            return $(this).attr("aria-valuenow") + "%";
        }
    )
});

window.addEventListener('keyup', function(event) {
    if (event.keyCode === 13) {
        document.getElementById('txtsize').innerHTML = "Loading...";
        document.getElementById('typesize').innerHTML = "";

        var txt = document.getElementById("form1").value;
        
        // POST
        fetch('/model', {
            headers: {
            'Content-Type': 'application/json'
            },

            method: 'POST',

            body: JSON.stringify({
                "text": txt
            })
        }).then(function (response) {
            return response.text();
        }).then(function (text) {
            text = JSON.parse(text);
            document.getElementById('txtsize').innerHTML = text.result;
            document.getElementById('typesize').innerHTML = text.typeExp;
            document.getElementById('progress-bar-compound').innerHTML = text.compound_sentiment+" %   ";
            document.getElementById('progress-bar-pos').innerHTML = text.pos_sentiment+" %   ";
            document.getElementById('progress-bar-neg').innerHTML = text.neg_sentiment+" %   ";
            document.getElementById('progress-bar-neu').innerHTML = text.neu_sentiment+" %   ";

            $('#progress-bar-compound').css("width",text.compound_sentiment+"%");
            $('#progress-bar-pos').css("width",text.pos_sentiment+"%");
            $('#progress-bar-neg').css("width",text.neg_sentiment+"%");
            $('#progress-bar-neu').css("width",text.neu_sentiment+"%");
        });
    }
});