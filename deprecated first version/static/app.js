let scene, camera, renderer, globe;
  renderer.setSize(window.innerWidth * 0.66, window.innerHeight);
  document.getElementById("globe").appendChild(renderer.domElement);

  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.position.set(5, 3, 5);
  scene.add(light);

  const geometry = new THREE.SphereGeometry(1, 64, 64);
  const texture = new THREE.TextureLoader().load(
    "https://cdn.jsdelivr.net/gh/visual-snow/earth-textures@main/2k_earth_daymap.jpg"
  );
  const material = new THREE.MeshStandardMaterial({ map: texture });

  globe = new THREE.Mesh(geometry, material);
  scene.add(globe);

  const res = await fetch("/cities");
  cities = await res.json();

  for (const city of cities) addCity(city);

  renderer.domElement.addEventListener("mousemove", onMouseMove);
  renderer.domElement.addEventListener("wheel", onWheel);
}

function latLonToVector3(lat, lon, radius = 1.01) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);

  const x = -radius * Math.sin(phi) * Math.cos(theta);
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);

  return new THREE.Vector3(x, y, z);
}

function addCity(city) {
  const pos = latLonToVector3(city.lat, city.lon);

  const dotGeom = new THREE.SphereGeometry(0.01, 8, 8);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0xff5555 });
  const dot = new THREE.Mesh(dotGeom, dotMat);
  dot.position.copy(pos);
  globe.add(dot);

  city.object = dot;
}

function animate() {
  requestAnimationFrame(animate);
  globe.rotation.y += 0.0008;
  renderer.render(scene, camera);
}

function onWheel(e) {
  camera.position.z += e.deltaY * 0.001;
  camera.position.z = Math.max(1.5, Math.min(5, camera.position.z));
}

function onMouseMove(e) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  const mouse = new THREE.Vector2(x, y);
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);

  const intersects = raycaster.intersectObject(globe);
  if (intersects.length > 0) {
    const p = intersects[0].point.clone().normalize();

    const lat = 90 - (Math.acos(p.y) * 180) / Math.PI;
    const lon = ((Math.atan2(p.z, p.x) * 180) / Math.PI) - 180;

    document.getElementById("coords").innerText =
      `lat: ${lat.toFixed(4)} lon: ${lon.toFixed(4)}`;
  }
}