import React from "react";
import { Link } from "react-router-dom";
import "../templates/Faq.css";

// Image imports - place the three provided images into src/assets/
// with names: img1.jpg, img2.jpg, img3.jpg
import img1 from "../assets/Elephant.jpeg";
import img2 from "../assets/Deer.jpeg";
import img3 from "../assets/Fox.jpeg";

const animals = [
  { name: "Elephant", desc: "Large and gentle giant", img: img1 },
  { name: "Deer", desc: "Graceful forest dweller", img: img2 },
  { name: "Fox", desc: "Cunning and curious", img: img3 },
  { name: "Boar", desc: "Forager of the underbrush", img: img2 },
  { name: "Owl", desc: "Nocturnal sentinel", img: img3 },
  { name: "Leopard", desc: "Silent and powerful", img: img1 },
  { name: "Hare", desc: "Quick and watchful", img: img3 },
  { name: "Badger", desc: "Steady ground forager", img: img2 },
  { name: "Eagle", desc: "Keen-eyed hunter", img: img3 },
  { name: "Wild Dog", desc: "Pack-oriented tracker", img: img1 },
  { name: "Heron", desc: "Wading water hunter", img: img2 },
  { name: "Porcupine", desc: "Well-armored loner", img: img3 },
  { name: "Tapir", desc: "Shy forest browser", img: img1 },
  { name: "Raccoon", desc: "Curious night visitor", img: img2 },
  { name: "Wolf", desc: "Coordinated pack hunter", img: img3 },
];

export default function Faq() {
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

          {/* <button className="cta-btn" aria-label="Create ticket">
            Create Ticket
          </button> */}
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
                  <button className="small-action">View</button>
                </div>
              </div>
            </article>
          ))}
        </section>
      </main>

      <footer className="faq-footer">
        <small>© {new Date().getFullYear()} Wildlife Surveillance • Real-time updates</small>
      </footer>
    </div>
  );
}