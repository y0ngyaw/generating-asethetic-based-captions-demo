var lock = false;

document.querySelector('input[type="file"]').addEventListener('change', function() {
	if (this.files && this.files[0]) {
		var img = document.querySelector('#input-image');  // $('img')[0]
		img.src = URL.createObjectURL(this.files[0]); // set src to blob url
		clearCaptions();

		var form_data = new FormData();
		form_data.append('file', this.files[0]);
		sendRequest(form_data)
	}
});

// $('#submit-btn').click(function() {
// 	var form_data = new FormData();
// 	var file = document.querySelector('#image-form')[0].files[0];
// 	form_data.append('file', file);

// 	sendRequest(form_data)
// })

$('.preview').click(function() {
	var img = document.querySelector('#input-image');
	img.src = this.src;
	clearCaptions();
	fetch(this.src)
	.then(res => res.blob())
	.then(blob => {
		const file = new File([blob], 'img.jpg', blob)
		var form_data = new FormData();
		form_data.append('file', file);

		sendRequest(form_data)
	})
})

function sendRequest(form_data) {
	if(lock === false){
		lock = true;
		document.querySelector('#overlay').className = 'overlay overlay-active';
		$.ajax({
			url: window.location.origin + '/image',
			type: 'POST',
			data: form_data,
			cache: false, 
			contentType: false,
	        processData: false,
			success: function(response) {
				document.querySelector('#overlay').className = 'overlay';
				if(response) {
					response.captions.forEach(appendCaption);
					lock = false;
				}
				else {
					console.log('Error');
				}
			}
		})
	}
}

var pos = 0;
$(window).bind('mousewheel DOMMouseScroll', function(event) {
	if (event.originalEvent.wheelDelta > 0 || event.originalEvent.detail < 0) {
		pos = pos + 50;
	} 
	else {
		if (pos > 1) {
			pos = pos - 50;
		}    
	}
	$('#scrollable-row').scrollLeft(pos)    
});

function appendCaption(caption, index) {
	parent = document.querySelector('#captions');

	div = document.createElement("div");
	div.className = "col";

	card = document.createElement("div");
	card.className = "card";
	card.style.animationDelay = 0.25*(index+1) + 's';

	card_body = document.createElement("div");
	card_body.className = "card-body";
	card_body.innerHTML = caption;

	card.append(card_body);
	div.append(card);
	parent.append(div);
}

function clearCaptions() {
	node = document.querySelector('#captions');
	while (node.firstChild) {
		node.removeChild(node.firstChild)
	}
}