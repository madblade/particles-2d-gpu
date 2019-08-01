var timeout;
var timeoutLimit = 300;
var isTryingToAnimate = false;

var slider = document.getElementById("particleslider");
var nbParticles = document.getElementById("numberparticles");
function sliderlistener() {
   nbParticles.innerText = slider.value;
   enforceParticleNumber = slider.value;
   tryToAnimate();
}
slider.oninput = sliderlistener;
slider.onchange = sliderlistener;
var enforceParticleNumber = 0;

function reinit() {
   enforceParticleNumber = 0;
   tryToAnimate();
}

function tryToAnimate() {
   clearTimeout(animate);
   if (!isTryingToAnimate) {
      isTryingToAnimate = true;
      timeout = setTimeout(animate, timeoutLimit);
   }
}

function getHeight(element) {
   var elementStyle = window.getComputedStyle(element);
   return element.offsetHeight +
       parseInt(elementStyle.marginTop, 10) +
       parseInt(elementStyle.marginBottom, 10);
}

function animate() {
   isTryingToAnimate = false;

   var element = document.getElementById('windy');
   var hr = document.getElementById('separator');
   var foot = document.getElementById('footer');
   var userinput = document.getElementById('userinput');
   var navHeight = getHeight(hr) + getHeight(foot) + getHeight(userinput);

   var winHeight = window.innerHeight;

   var paddingBottom = 20;
   var width = element.offsetWidth;
   var height = winHeight - navHeight - paddingBottom;
   element.style.height = height.toString();

   var nbSamples =
       Math.floor(Math.min(25 * width * height / 500, 30000));

   if (enforceParticleNumber > 0) nbSamples = enforceParticleNumber;

   slider.value = nbSamples;
   nbParticles.innerText = nbSamples.toString();

   var gl = null;

   // Firefox, Chrome
   if (!gl) {
      element.getContext('webgl', { antialiasing: true });
   }

   // IE
   if (!gl) {
      try {
         gl = element.getContext('experimental-webgl');
      } catch (error) {
         var msg = '[Dashboard] Error while creating WebGL context: ' + error.toString();
         throw Error(msg);
      }
   }

   // Unsupported
   if (!gl) {
      if (!document.getElementById('webgl-unsupported')) {
         var image = document.createElement('div');
         image.setAttribute('id', 'webgl-unsupported');
         image.setAttribute('style', 'position: absolute;left:0;');
         image.innerText = 'Your browser does not support WebGL :/';
         document.getElementById('pusher').appendChild(image);
      }
      return;
   }

   Windy.start(
       gl, element, width, height, nbSamples, null
   );
}

window.addEventListener('DOMContentLoaded', reinit);
window.addEventListener('resize', reinit);
