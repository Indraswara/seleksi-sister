const http = require('http');

function sendRequest(options, postData = null) {
    return new Promise((resolve, reject) => {
        const req = http.request(options, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                console.log(`Response received:\n${data}`);
                resolve(data);
            });
        });

        req.on('error', (e) => {
            console.error(`Problem with request: ${e.message}`);
            reject(e);
        });

        if (postData) {
            req.write(postData);
        }

        req.end();
    });
}

async function main() {
    // Example GET request
    const getOptions = {
        hostname: 'localhost',
        port: 8080,
        path: '/nilai-akhir',
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
    };

    console.log('Sending GET request...');
    await sendRequest(getOptions);

    // Example POST request
    const postData = JSON.stringify({ bjir: 'okeh' });
    const postOptions = {
        hostname: 'localhost',
        port: 8080,
        path: '/submit',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData),
        },
    };

    console.log('Sending POST request...');
    await sendRequest(postOptions, postData);
}

main().catch((err) => {
    console.error(`Error: ${err.message}`);
});