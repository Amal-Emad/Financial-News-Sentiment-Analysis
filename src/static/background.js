const canvas = document.getElementById("moneyCanvas");
const ctx = canvas.getContext("2d");

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

let dollarSigns = [];

class Dollar {
    constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * -canvas.height;
        this.size = Math.random() * 30 + 10;
        this.speed = Math.random() * 3 + 1;
    }

    update() {
        this.y += this.speed;
        if (this.y > canvas.height) {
            this.y = Math.random() * -canvas.height;
            this.x = Math.random() * canvas.width;
        }
    }

    draw() {
        ctx.font = `${this.size}px Arial`;
        ctx.fillStyle = "green";
        ctx.fillText("$", this.x, this.y);
    }
}

function init() {
    for (let i = 0; i < 50; i++) {
        dollarSigns.push(new Dollar());
    }
}

function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    dollarSigns.forEach(dollar => {
        dollar.update();
        dollar.draw();
    });
    requestAnimationFrame(animate);
}

window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

init();
animate();
