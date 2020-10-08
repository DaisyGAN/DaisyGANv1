<?php
    $token = '<BOT TOKEN>';
    $j = json_decode(file_get_contents("php://input"));
    if(isset($j->{'message'}->{'text'}))
    {
        if(strstr($j->{'message'}->{'text'}, "/isdaisy") != FALSE && isset($j->{'message'}->{'reply_to_message'}->{'text'}))
        {
            $percent = rtrim(shell_exec('/srv/cfdgan "' . $j->{'message'}->{'reply_to_message'}->{'text'} . '"'), "\n");
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode('"'.$j->{'message'}->{'reply_to_message'}->{'text'}).'"'."+is+" . urlencode($percent) . "%25+something+Daisy+(@VXF97)+would+say.");
            http_response_code(200);
            exit;
        }
        if(strstr($j->{'message'}->{'text'}, "/quote") != FALSE)
        {
            $file = file("kds.txt"); 
            $line = $file[rand(0, count($file) - 1)];
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&parse_mode=HTML&text=" . urlencode("<a href=\"tg://user?id=1155563040\">Daisy:</a> ") .urlencode($line));
            http_response_code(200);
            exit;
        }
    }
    http_response_code(200);
?>
