function toggleSection(element) {
    const allContents = document.querySelectorAll('.section-content');
    allContents.forEach(content => {
        if (content !== element.nextElementSibling) {
            content.classList.remove('active');
        }
    });

    const content = element.nextElementSibling;
    content.classList.toggle('active');
}