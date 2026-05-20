import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { ArrowRight, Activity, Database, Zap, Sparkles } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

// -----------------------------------------------------------------------------
// A. NAVBAR — "The Floating Island"
// -----------------------------------------------------------------------------
const Navbar = () => {
  const navRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        start: 'top -50',
        end: 99999,
        toggleClass: {
          className: 'bg-background/80 backdrop-blur-xl border-primary/10 text-primary',
          targets: navRef.current
        },
        onToggle: (self) => {
          if (!self.isActive) {
            navRef.current.classList.add('text-background', 'border-transparent');
            navRef.current.classList.remove('bg-background/80', 'backdrop-blur-xl', 'border-primary/10', 'text-primary');
          } else {
            navRef.current.classList.remove('text-background', 'border-transparent');
          }
        }
      });
    }, navRef);
    return () => ctx.revert();
  }, []);

  return (
    <nav 
      ref={navRef}
      className="fixed top-6 left-1/2 -translate-x-1/2 z-50 flex items-center justify-between px-6 py-3 rounded-full transition-all duration-500 w-[90%] max-w-5xl text-background border border-transparent"
    >
      <div className="font-display font-bold text-lg tracking-tight">Amazon RecSys</div>
      <div className="hidden md:flex items-center space-x-8 font-mono text-sm">
        <a href="#features" className="hover:-translate-y-px transition-transform">Features</a>
        <a href="#philosophy" className="hover:-translate-y-px transition-transform">Philosophy</a>
        <a href="#protocol" className="hover:-translate-y-px transition-transform">Protocol</a>
      </div>
      <button className="magnetic-button bg-accent text-white px-5 py-2 rounded-full font-sans text-sm font-medium">
        Get Recommendations
      </button>
    </nav>
  );
};

// -----------------------------------------------------------------------------
// B. HERO SECTION — "The Opening Shot"
// -----------------------------------------------------------------------------
const Hero = () => {
  const sectionRef = useRef(null);
  
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.hero-el', {
        y: 40,
        opacity: 0,
        duration: 1.2,
        stagger: 0.08,
        ease: 'power3.out',
        delay: 0.2
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section ref={sectionRef} className="relative h-[100dvh] w-full flex items-end pb-24 md:pb-32 px-6 md:px-16 overflow-hidden">
      {/* Background Image with Gradient Overlay */}
      <div 
        className="absolute inset-0 z-0 bg-cover bg-center"
        style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?q=80&w=2000&auto=format&fit=crop)' }}
      />
      <div className="absolute inset-0 z-10 bg-gradient-to-t from-primary via-primary/80 to-transparent" />
      
      {/* Content */}
      <div className="relative z-20 w-full max-w-4xl text-background">
        <h1 className="flex flex-col mb-8">
          <span className="hero-el font-display font-bold text-3xl md:text-5xl tracking-tight mb-2">Recommendation is the</span>
          <span className="hero-el text-drama text-6xl md:text-8xl leading-none">Future.</span>
        </h1>
        <p className="hero-el font-sans text-lg md:text-xl max-w-xl mb-10 text-background/80">
          Advanced product recommendation engine powered by machine learning.
        </p>
        <div className="hero-el">
          <button className="magnetic-button group bg-accent text-white px-8 py-4 rounded-full font-sans font-medium text-lg flex items-center space-x-3">
            <span>Get Recommendations</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </div>
    </section>
  );
};

// -----------------------------------------------------------------------------
// C. FEATURES — "Interactive Functional Artifacts"
// -----------------------------------------------------------------------------
const Features = () => {
  // Card 1: Diagnostic Shuffler
  const [shuffleItems, setShuffleItems] = useState([
    { id: 1, text: "Sequential Attention" },
    { id: 2, text: "User History Analysis" },
    { id: 3, text: "Temporal Dynamics" }
  ]);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setShuffleItems(prev => {
        const newItems = [...prev];
        const last = newItems.pop();
        newItems.unshift(last);
        return newItems;
      });
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // Card 2: Telemetry Typewriter
  const [text, setText] = useState("");
  const fullText = "> Processing debiased ranking... \n> Normalizing propensity scores...\n> Final recommendations ready.";
  
  useEffect(() => {
    let i = 0;
    const interval = setInterval(() => {
      setText(fullText.substring(0, i));
      i++;
      if (i > fullText.length) i = 0; // Loop for effect
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <section id="features" className="py-32 px-6 md:px-16 bg-surface text-primary">
      <div className="max-w-7xl mx-auto">
        <h2 className="font-display font-bold text-4xl mb-16 text-center">Core Mechanics</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          
          {/* Card 1: Shuffler */}
          <div className="bg-background rounded-[2rem] p-8 shadow-sm border border-primary/5 flex flex-col h-96">
            <div className="mb-auto">
              <Activity className="text-accent mb-4" />
              <h3 className="font-sans font-bold text-xl mb-2">Sequential Attention</h3>
              <p className="text-primary/70 text-sm">Tracking temporal user behavior.</p>
            </div>
            <div className="relative h-40 flex items-end justify-center">
              {shuffleItems.map((item, i) => (
                <div 
                  key={item.id}
                  className="absolute w-full bg-surface text-primary font-mono text-xs p-4 rounded-xl shadow-md border border-primary/10 transition-all duration-700"
                  style={{
                    transform: `translateY(-${i * 15}px) scale(${1 - i * 0.05})`,
                    opacity: 1 - i * 0.2,
                    zIndex: 10 - i,
                    transitionTimingFunction: 'cubic-bezier(0.34, 1.56, 0.64, 1)'
                  }}
                >
                  {item.text}
                </div>
              ))}
            </div>
          </div>

          {/* Card 2: Typewriter */}
          <div className="bg-background rounded-[2rem] p-8 shadow-sm border border-primary/5 flex flex-col h-96">
            <div className="mb-auto">
              <div className="flex items-center justify-between mb-4">
                <Database className="text-accent" />
                <div className="flex items-center space-x-2 bg-surface px-3 py-1 rounded-full">
                  <div className="w-2 h-2 bg-accent rounded-full animate-pulse" />
                  <span className="font-mono text-xs">Live Feed</span>
                </div>
              </div>
              <h3 className="font-sans font-bold text-xl mb-2">Debiased Ranking</h3>
              <p className="text-primary/70 text-sm">Correcting for popularity bias.</p>
            </div>
            <div className="bg-dark text-background p-4 rounded-xl h-40 overflow-hidden font-mono text-sm font-light">
              <div className="whitespace-pre-line">{text}<span className="inline-block w-2 h-4 bg-accent ml-1 animate-pulse"/></div>
            </div>
          </div>

          {/* Card 3: Scheduler / Visualizer */}
          <div className="bg-background rounded-[2rem] p-8 shadow-sm border border-primary/5 flex flex-col h-96">
            <div className="mb-auto">
              <Zap className="text-accent mb-4" />
              <h3 className="font-sans font-bold text-xl mb-2">Vector Search</h3>
              <p className="text-primary/70 text-sm">Sub-second nearest neighbor retrieval.</p>
            </div>
            <div className="h-40 bg-surface rounded-xl flex items-center justify-center relative overflow-hidden group">
              <div className="grid grid-cols-5 gap-2 opacity-50">
                {Array.from({length: 15}).map((_, i) => (
                  <div key={i} className="w-4 h-4 rounded-sm bg-primary/20 group-hover:bg-accent/40 transition-colors delay-[${i*50}ms]" />
                ))}
              </div>
              {/* Fake cursor animation */}
              <div className="absolute w-4 h-4 bg-accent rounded-full opacity-0 group-hover:animate-[ping_2s_cubic-bezier(0,0,0.2,1)_infinite]" />
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

// -----------------------------------------------------------------------------
// D. PHILOSOPHY — "The Manifesto"
// -----------------------------------------------------------------------------
const Philosophy = () => {
  const sectionRef = useRef(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('.phil-word', {
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 60%',
        },
        y: 30,
        opacity: 0,
        stagger: 0.1,
        duration: 1,
        ease: 'power3.out'
      });
    }, sectionRef);
    return () => ctx.revert();
  }, []);

  return (
    <section id="philosophy" ref={sectionRef} className="relative py-40 px-6 md:px-16 bg-dark text-background overflow-hidden">
      {/* Background Texture */}
      <div 
        className="absolute inset-0 z-0 opacity-20 bg-cover bg-fixed bg-center"
        style={{ backgroundImage: 'url(https://images.unsplash.com/photo-1518531933037-91b2f5f229cc?q=80&w=2000&auto=format&fit=crop)' }}
      />
      
      <div className="relative z-10 max-w-5xl mx-auto flex flex-col space-y-16">
        <p className="font-sans text-xl md:text-2xl text-background/60 max-w-2xl phil-word">
          Most e-commerce focuses on: generic popularity and basic collaborative filtering.
        </p>
        <h2 className="text-drama text-5xl md:text-7xl leading-tight max-w-4xl">
          <span className="phil-word inline-block mr-3">We</span>
          <span className="phil-word inline-block mr-3">focus</span>
          <span className="phil-word inline-block mr-3">on:</span>
          <span className="phil-word inline-block text-accent">precision</span>
          <span className="phil-word inline-block mr-3">learning</span>
          <span className="phil-word inline-block mr-3">powered</span>
          <span className="phil-word inline-block mr-3">by</span>
          <span className="phil-word inline-block mr-3">biological-like</span>
          <span className="phil-word inline-block">data.</span>
        </h2>
      </div>
    </section>
  );
};

// -----------------------------------------------------------------------------
// E. PROTOCOL — "Sticky Stacking Archive"
// -----------------------------------------------------------------------------
const Protocol = () => {
  const containerRef = useRef(null);
  const steps = [
    { num: '01', title: 'Data Ingestion', desc: 'Continuous processing of user interactions and item metadata.' },
    { num: '02', title: 'Graph Construction', desc: 'Building complex relationship networks between users and products.' },
    { num: '03', title: 'Inference', sub: 'Sub-second FAISS retrieval.', desc: 'Generating Top-K recommendations dynamically.' }
  ];

  useEffect(() => {
    const ctx = gsap.context(() => {
      const cards = gsap.utils.toArray('.stack-card');
      
      cards.forEach((card, index) => {
        if (index === cards.length - 1) return; // Skip last card
        
        ScrollTrigger.create({
          trigger: card,
          start: 'top top',
          pin: true,
          pinSpacing: false,
          endTrigger: containerRef.current,
          end: 'bottom bottom',
          animation: gsap.to(card, {
            scale: 0.9,
            opacity: 0.5,
            filter: 'blur(20px)',
            ease: 'none'
          }),
          scrub: true
        });
      });
    }, containerRef);
    return () => ctx.revert();
  }, []);

  return (
    <section id="protocol" ref={containerRef} className="bg-background relative">
      {steps.map((step, i) => (
        <div key={i} className="stack-card h-screen w-full sticky top-0 bg-background flex items-center justify-center p-6 md:p-16 border-t border-primary/5">
          <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            
            {/* Visualizer Side */}
            <div className="h-64 md:h-96 bg-surface rounded-[3rem] flex items-center justify-center overflow-hidden relative">
              {i === 0 && (
                <div className="w-32 h-32 border-4 border-accent/20 rounded-full animate-[spin_10s_linear_infinite] flex items-center justify-center">
                  <div className="w-16 h-16 border-4 border-primary/40 rounded-full animate-[spin_5s_linear_infinite_reverse]" />
                </div>
              )}
              {i === 1 && (
                <div className="w-full h-full bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-accent/10 to-transparent relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-full h-[2px] bg-accent/40 animate-[bounce_2s_infinite]" />
                  </div>
                </div>
              )}
              {i === 2 && (
                <svg viewBox="0 0 100 50" className="w-3/4 stroke-accent fill-none stroke-2">
                  <path d="M0 25 L30 25 L40 10 L60 40 L70 25 L100 25" strokeDasharray="200" strokeDashoffset="200">
                    <animate attributeName="stroke-dashoffset" values="200;0" dur="2s" repeatCount="indefinite" />
                  </path>
                </svg>
              )}
            </div>

            {/* Text Side */}
            <div>
              <div className="font-mono text-accent text-xl mb-4">Step {step.num}</div>
              <h2 className="font-display font-bold text-4xl md:text-5xl mb-6 text-primary">{step.title}</h2>
              <p className="font-sans text-lg text-primary/70">{step.desc}</p>
            </div>

          </div>
        </div>
      ))}
    </section>
  );
};

// -----------------------------------------------------------------------------
// F. GET STARTED & FOOTER
// -----------------------------------------------------------------------------
const Footer = () => {
  return (
    <>
      <section className="py-32 px-6 md:px-16 bg-surface text-center">
        <h2 className="text-drama text-5xl md:text-6xl text-primary mb-8">Ready to experience precision?</h2>
        <button className="magnetic-button bg-accent text-white px-10 py-5 rounded-full font-sans font-bold text-lg inline-flex items-center space-x-3">
          <Sparkles className="w-5 h-5" />
          <span>Get Recommendations</span>
        </button>
      </section>

      <footer className="bg-dark text-background pt-20 pb-10 px-6 md:px-16 rounded-t-[4rem] -mt-10 relative z-10">
        <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-12 mb-16">
          <div className="md:col-span-2">
            <h3 className="font-display font-bold text-2xl mb-4">Amazon RecSys</h3>
            <p className="font-sans text-background/60 max-w-sm">
              Advanced product recommendation engine powered by machine learning.
            </p>
          </div>
          <div>
            <h4 className="font-mono text-xs uppercase tracking-widest text-background/40 mb-6">Navigation</h4>
            <ul className="space-y-3 font-sans text-sm">
              <li><a href="#" className="hover:text-accent transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-accent transition-colors">Philosophy</a></li>
              <li><a href="#" className="hover:text-accent transition-colors">Protocol</a></li>
            </ul>
          </div>
          <div>
            <h4 className="font-mono text-xs uppercase tracking-widest text-background/40 mb-6">System</h4>
            <div className="flex items-center space-x-3 bg-white/5 rounded-xl p-4 border border-white/10">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_rgba(34,197,94,0.5)]" />
              <span className="font-mono text-xs">System Operational</span>
            </div>
          </div>
        </div>
        <div className="max-w-7xl mx-auto border-t border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center text-xs text-background/40 font-mono">
          <p>© 2026 Amazon RecSys</p>
          <div className="space-x-4 mt-4 md:mt-0">
            <a href="#" className="hover:text-background transition-colors">Privacy</a>
            <a href="#" className="hover:text-background transition-colors">Terms</a>
          </div>
        </div>
      </footer>
    </>
  );
};

// -----------------------------------------------------------------------------
// MAIN APP
// -----------------------------------------------------------------------------
function App() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <Hero />
      <Features />
      <Philosophy />
      <Protocol />
      <Footer />
    </div>
  );
}

export default App;
