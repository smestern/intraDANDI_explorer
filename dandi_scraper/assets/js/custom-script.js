function tableDropdowncust() {
    console.log('adding effects')
    $("#datatable-row-ids tr>td:first-of-type").unbind()
    $("#datatable-row-ids tr>td:first-of-type").on("click",
        function() {
            row = $(this).closest('tr')
            if ($(row).hasClass("expanded")) {
                $(".extraInfo").remove()
                $(".expanded").removeClass("expanded")
            } else {
                $(".extraInfo").remove()
                $(".expanded").removeClass("expanded")
                $(row).addClass("expanded")
            }
            if ($(row).hasClass('expanded')) {
                info = ''
                $(row).find("td:nth-child(n+8)").each(function() {
                    info += $(this).attr('data-dash-column') + ": "+ $(this).text() +'\n'
                })
                $(row).after('<tr type="smn" id="smn_'+ $(row).find($( "td[data-dash-column='specimen_id']" )).text() +'" class="extraInfo"><td colspan="3" style="width: 100%; max-width:100%"><pre>'
                +info + '</pre></td></tr>')
                //emit custom event
               
            }
        }
    )
}

window.fetch = new Proxy(window.fetch, {
    apply(fetch, that, args) {
        // Forward function call to the original fetch
        const result = fetch.apply(that, args);

        // Do whatever you want with the resulting Promise
        result.then((response) => {
            if (args[0] == '/_dash-update-component') {
                setTimeout(function() {tableDropdowncust()}, 1000)
            }})
        return result
        }
    }
    )

$(document).ready(function() {
    setTimeout(function() {tableDropdowncust()}, 1000)
})