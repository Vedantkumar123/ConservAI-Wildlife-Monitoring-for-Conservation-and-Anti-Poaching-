import React, { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "../templates/Faq.css";

import img1 from "../assets/Elephant.jpeg";
import img2 from "../assets/Deer.jpeg";
import img3 from "../assets/Fox.jpeg";
import img4 from "../assets/Boar.jpg";
import img5 from "../assets/Owl.jpg";
import img6 from "../assets/Leopard.webp";
import img7 from "../assets/Hare.webp";
import img8 from "../assets/Badger.webp";
import img9 from "../assets/Eagle.webp";
import img10 from "../assets/WildDog.webp";
import img11 from "../assets/Heron.webp";
import img12 from "../assets/Porcupine.webp";
import img13 from "../assets/Tapir.webp";
import img14 from "../assets/Raccoon.webp";
import img15 from "../assets/Wolf.webp";
import img16 from "../assets/Lion.webp";
import img17 from "../assets/Tiger.webp";
import img18 from "../assets/Giraffe.webp";

const animals = [
  { name: "Elephant", desc: "Large and gentle giant", img: img1 },
  { name: "Deer", desc: "Graceful forest dweller", img: img2 },
  { name: "Fox", desc: "Cunning and curious", img: img3 },
  { name: "Boar", desc: "Forager of the underbrush", img: img4 },
  { name: "Owl", desc: "Nocturnal sentinel", img: img5 },
  { name: "Leopard", desc: "Silent and powerful", img: img6 },
  { name: "Hare", desc: "Quick and watchful", img: img7 },
  { name: "Badger", desc: "Steady ground forager", img: img8 },
  { name: "Eagle", desc: "Keen-eyed hunter", img: img9 },
  { name: "Wild Dog", desc: "Pack-oriented tracker", img: img10 },
  { name: "Heron", desc: "Wading water hunter", img: img11 },
  { name: "Porcupine", desc: "Well-armored loner", img: img12 },
  { name: "Tapir", desc: "Shy forest browser", img: img13 },
  { name: "Raccoon", desc: "Curious night visitor", img: img14 },
  { name: "Wolf", desc: "Coordinated pack hunter", img: img15 },
  { name: "Lion", desc: "Majestic king of the jungle", img: img16 },
  { name: "Tiger", desc: "Stealthy striped predator", img: img17 },
  { name: "Giraffe", desc: "Tall and gentle browser", img: img18 },
];

export default function Faq() {
  const [selectedImg, setSelectedImg] = useState(null);

  // Close modal with Esc key
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === "Escape") setSelectedImg(null);
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, []);

  return (
    <div className="faq-app">
      <nav className="faq-nav">
        <div className="nav-left">
          <h1 className="brand">Wildlife</h1>
          <span className="subbrand">Surveillance FAQ</span>
        </div>
        <div className="nav-right">
          <Link to="/" className="cta-btn" aria-label="Home">
            Home
          </Link>
        </div>
      </nav>

      <main className="faq-main">
        <section className="intro">
          <h2>Frequently Observed Species</h2>
          <p>
            Browse recent sightings and commonly detected animals. Tap a card for
            more details or to raise an alert.
          </p>
        </section>

        <section className="card-grid" role="list">
          {animals.map((a, idx) => (
            <article className="animal-card" key={idx} role="listitem">
              <div
                className="card-image"
                style={{ backgroundImage: `url(${a.img})` }}
                aria-hidden="true"
              />
              <div className="card-body">
                <h3 className="animal-name">{a.name}</h3>
                <p className="animal-desc">{a.desc}</p>
                <div className="card-meta">
                  <span className="tag">Recent</span>
                  <button
                    className="small-action"
                    onClick={() => setSelectedImg(a.img)}
                  >
                    View
                  </button>
                </div>
              </div>
            </article>
          ))}
        </section>
      </main>

      <footer className="faq-footer">
        <small>
          © {new Date().getFullYear()} Wildlife Surveillance • Real-time updates
        </small>
      </footer>

      {/* Fullscreen Modal */}
      {selectedImg && (
        <div
          className="fullscreen-overlay"
          onClick={() => setSelectedImg(null)}
        >
          <img
            src={selectedImg}
            alt="Expanded animal"
            className="fullscreen-img"
            onClick={(e) => e.stopPropagation()} // Prevent closing on image click
          />
        </div>
      )}
    </div>
  );
}
