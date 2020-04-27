$(document).ready(function(){
    let file;
    const input = document.querySelector("input#input-image");
    $("#input-image").change( function(e) {
        var reader = new FileReader();
        reader.onload = function (e) {
          $("#show-image").attr('src', e.target.result)
        };
        reader.readAsDataURL(e.target.files[0]);
    })

    $("#get-breed").click( function (e) {
        uploadImages(input.files[0])
    })

});

function uploadImages(file){
  var formData = new FormData();
  formData.append('file',file);

  $.ajax({
    url:'/upload-image',
    data:formData,
    type:'POST',
    cache:false,
    contentType:false,
    processData:false,
    // beforeSend: function() {
    //   document.querySelector(".btn-upload").innerText = "Uploading...";
    //   document.querySelector(".btn-upload").disabled = true;
    // },
    success:function(r){
      console.log(r);
    },
    failure: function(e) {
      console.log(e)
    }
})
}