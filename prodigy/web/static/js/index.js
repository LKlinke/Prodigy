$(document).ready(function () {
    $(".spinner").hide();
});

function hide_results() {
    $("#result_equiv").hide();
    $("#result_analysis").hide();
    $("#result_playground").hide();
}

function get_message(response) {
    if (response.statusText != "timeout")
        return response.responseJSON.message
    return response.statusText
}

function equivalence_request() {
    $('.submit_button').fadeOut('fast');
    let myform = document.getElementById("equiv_check");
    let fd = new FormData(myform);
    $(".spinner").show();
    hide_results();
    $.ajax({
        url: "/equivalence",
        data: fd,
        cache: false,
        processData: false,
        contentType: false,
        type: 'POST',
        success: function (response, status) {
            try {
                $(".spinner").hide();
                if (response.equivalent) {
                    $("#result_equiv").text("Program Equivalence Verified").css({"color": "green"}).show();
                } else {
                    $("#result_equiv").text("Programs are not equivalent").css({"color": "red"}).show();
                }
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        error: function (response, status) {
            try {
                $(".spinner").hide();
                $("#result_equiv").text(get_message(response)).css({"color": "red"}).show();
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        timeout: 30000
    });
}

function analysis_request() {
    $('.submit_button').fadeOut('fast');
    let myform = document.getElementById("distribution_transform");
    let fd = new FormData(myform);
    $(".spinner").show();
    hide_results();
    $.ajax({
        url: "/analyze",
        data: fd,
        cache: false,
        processData: false,
        contentType: false,
        type: 'POST',
        success: function (response, status) {
            try {
                $(".spinner").hide();
                $("#result_analysis").val(response.distribution).css({"color": "black"}).show();
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        error: function (response, status) {
            try {
                $(".spinner").hide();
                $("#result_analysis").val(get_message(response)).css({"color": "red"}).show();
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        timeout: 30000
    });
}

function playground_request() {
    $('.submit_button').fadeOut('fast');
    let myform = document.getElementById("playground");
    let fd = new FormData(myform);
    $(".spinner").show();
    hide_results();
    $.ajax({
        url: "/playground",
        data: fd,
        cache: false,
        processData: false,
        contentType: false,
        type: 'POST',
        success: function (response, status) {
            try {
                $(".spinner").hide();
                $("#result_playground").val(response.distribution).css({"color": "black"}).show();
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        error: function (response, status) {
            try {
                $(".spinner").hide();
                $("#result_playground").val(response.responseJSON.message).css({"color": "red"}).show();
            } finally {
                $(".submit_button").fadeIn();
            }
        },
        timeout: 30000
    });
}
