document.addEventListener("DOMContentLoaded", function () {
    var settingsBtn = document.querySelector(".settings-btn");
    var aboutMeSection = document.querySelector(".about-me");
    var changePasswordScreen = document.querySelector(".change-password-screen");

    settingsBtn.addEventListener("click", function (event) {
        event.preventDefault();
        aboutMeSection.style.display = "none";
        changePasswordScreen.style.display = "block";
    });
});
