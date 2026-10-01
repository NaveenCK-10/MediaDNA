// MediaDNA Forensic UI Interactions
document.addEventListener('DOMContentLoaded', () => {
    console.log('[SYSTEM ONLINE] MediaDNA Frontend Analytics Loaded');
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });
});
