$(document).ready(function () {
    $(".spinner").hide();
});

function hide_results() {
    $("#result_equiv").hide();
    $("#result_analysis").hide();
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
            $(".spinner").hide();
            if (response.equivalent) {
                $("#result_equiv").text("Program Equivalence Verified").css({"color": "green"}).show();
            } else {
                $("#result_equiv").text("Programs are not equivalent").css({"color": "red"}).show();
            }
            $(".submit_button").fadeIn();
        },
        error: function (response, status) {
            $(".spinner").hide();
            $("#result_equiv").text(response.statusText).css({"color": "red"}).show();
            $(".submit_button").fadeIn();
        },
        timeout: 20000
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
            $(".spinner").hide();
            $("#result_analysis").text(response.distribution).show();
            $(".submit_button").fadeIn();
        },
        error: function (response, status) {
            $(".spinner").hide();
            $("#result_analysis").text(response.statusText).css({"color": "red"}).show();
            $(".submit_button").fadeIn();
        },
        timeout: 20000
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
            $(".spinner").hide();
            $("#result_playground").text(response.distribution).show();
            $(".submit_button").fadeIn();
        },
        error: function (response, status) {
            $(".spinner").hide();
            $("#result_playground").text(response.statusText).css({"color": "red"}).show();
            $(".submit_button").fadeIn();
        },
        timeout: 20000
    });
}