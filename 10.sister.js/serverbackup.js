const net = require('net');
const url = require('url');
const querystring = require('querystring');
const fs = require('fs');
const NilaiController = require('./NilaiController');

// Middleware array
const middlewares = [];

const routes = {
  GET: {},
  POST: {},
  PUT: {},
  DELETE: {}
};

function use(middleware) {
  middlewares.push(middleware);
}

function addRoute(method, path, handler) {
  const parts = path.split('/').filter(Boolean);
  let current = routes[method];

  parts.forEach((part, i) => {
    if (!current[part]) {
      current[part] = {};
    }
    current = current[part];

    if (i === parts.length - 1) {
      current['__handler'] = handler;
    }
  });
}

function matchRoute(method, pathname) {
  const parts = pathname.split('/').filter(Boolean);
  let current = routes[method];
  const params = {};

  for (let part of parts) {
    if (current[part]) {
      current = current[part];
    } else {
      return null;
    }
  }

  return current['__handler'] ? { handler: current['__handler'], params } : null;
}

function parseRequest(reqLine) {
  const [method, pathWithQuery] = reqLine.split(' ');
  const { pathname, query } = url.parse(pathWithQuery, true);
  return { method, pathname, query };
}

function handleRequest(req, res) {
  const { method, pathname, query } = req;
  const route = matchRoute(method, pathname);

  if (route) {
    req.query = query;
    req.params = route.params;

    res.send = (statusCode, body) => {
      res.write(`HTTP/1.1 ${statusCode}\r\n`);
      res.write('Content-Type: application/json\r\n');
      res.write('\r\n');
      res.write(JSON.stringify(body));
      res.end();
    };

    // Execute middleware
    let index = 0;

    function next() {
      if (index < middlewares.length) {
        middlewares[index++](req, res, next);
      } else {
        route.handler(req, res);
      }
    }

    next();
  } else {
    res.write('HTTP/1.1 404 Not Found\r\n');
    res.write('Content-Type: text/plain\r\n');
    res.write('\r\n');
    res.write('404 Not Found');
    res.end();
  }
}

function handleConnection(socket) {
  socket.on('data', (data) => {
    const request = data.toString().split('\r\n');
    const headers = request.slice(1, request.indexOf(''));
    const body = request.slice(request.indexOf('') + 1).join('\r\n');
    const reqLine = request[0];

    const res = {
      write: socket.write.bind(socket),
      end: socket.end.bind(socket)
    };

    const { method, pathname, query } = parseRequest(reqLine);

    const req = {
      method,
      pathname,
      query,
      headers: headers.reduce((acc, header) => {
        const [key, value] = header.split(': ');
        acc[key.toLowerCase()] = value;
        return acc;
      }, {}),
      body: ''
    };

    if (method === 'POST' || method === 'PUT') {
      const contentType = req.headers['content-type'];
      if (contentType === 'application/json') {
        req.body = JSON.parse(body);
      } else if (contentType === 'application/x-www-form-urlencoded') {
        req.body = querystring.parse(body);
      } else if (contentType.startsWith('multipart/form-data')) {
        const boundary = contentType.split('boundary=')[1];
        req.body = parseMultipartFormData(body, boundary);
      } else {
        req.body = body;
      }
    }

    handleRequest(req, res);
  });
}

function parseMultipartFormData(body, boundary) {
  const parts = body.split(`--${boundary}`).filter(part => part && part !== '--');
  const result = {};

  parts.forEach(part => {
    const [rawHeaders, rawBody] = part.split('\r\n\r\n');
    const headers = rawHeaders.split('\r\n').reduce((acc, header) => {
      const [key, value] = header.split(': ');
      acc[key.toLowerCase()] = value;
      return acc;
    }, {});
    const nameMatch = headers['content-disposition'].match(/name="([^"]+)"/);
    if (nameMatch) {
      const name = nameMatch[1];
      result[name] = rawBody.trim();
    }
  });

  return result;
}

const server = net.createServer(handleConnection);

// Sample middleware to log requests
use((req, res, next) => {
  console.log(`${req.method} ${req.pathname}`);
  next();
});

addRoute('GET', '/nilai-akhir', NilaiController.getNilaiAkhir);
addRoute('POST', '/submit/a', NilaiController.postData);
addRoute('PUT', '/update/a', NilaiController.putData);
addRoute('DELETE', '/delete/a', NilaiController.deleteData);

server.listen(3000, () => {
  console.log('Server is running on port 3000');
});
